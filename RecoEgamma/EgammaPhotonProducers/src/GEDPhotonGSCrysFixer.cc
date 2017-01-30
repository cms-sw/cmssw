#include <vector>
#include <memory>

#include "RecoEgamma/EgammaPhotonProducers/interface/GEDPhotonGSCrysFixer.h"

#include "FWCore/Utilities/interface/isFinite.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/EgammaCandidates/interface/PhotonCore.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"

#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionBaseClass.h" 
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterFunctionFactory.h" 
#include "RecoEgamma/EgammaTools/interface/GainSwitchTools.h" 

typedef edm::ValueMap<reco::PhotonRef> PhotonRefMap;

namespace {
  reco::PhotonCoreRef
  getNewCore(reco::Photon const& oldPho, edm::Handle<reco::PhotonCoreCollection> const& newCores, edm::ValueMap<reco::SuperClusterRef> const& scMap)
  {
    for (unsigned iC(0); iC != newCores->size(); ++iC) {
      reco::PhotonCoreRef newCore(newCores, iC);
      auto&& oldSCRef(scMap[newCore]);
      if (oldSCRef.isNonnull() && oldSCRef == oldPho.superCluster())
	return newCore;
    }
    return reco::PhotonCoreRef(newCores.id());
  }
}


GEDPhotonGSCrysFixer::GEDPhotonGSCrysFixer(const edm::ParameterSet& config) :
  energyCorrector_(config, consumesCollector())
{
  getToken(inputPhotonsToken_, config, "photons");
  getToken(newCoresToken_, config, "newCores");
  getToken(newCoresToOldSCMapToken_, config, "newCores");
  getToken(ebHitsToken_, config, "barrelEcalHits");
  getToken(verticesToken_, config, "primaryVertexProducer");

  if(config.existsAs<edm::ParameterSet>("regressionConfig")) {
    auto&& collector(consumesCollector());
    energyCorrector_.gedRegression()->setConsumes(collector);
  }

  // output photon collection
  produces<reco::PhotonCollection>();
  // new photon -> old photon
  produces<PhotonRefMap>();
}

GEDPhotonGSCrysFixer::~GEDPhotonGSCrysFixer() 
{
}

void
GEDPhotonGSCrysFixer::beginLuminosityBlock(edm::LuminosityBlock const& _lb, edm::EventSetup const& _setup) 
{
  energyCorrector_.init(_setup);

  edm::ESHandle<CaloGeometry> geometryHandle;
  _setup.get<CaloGeometryRecord>().get(geometryHandle);
  geometry_ = geometryHandle.product();

  edm::ESHandle<CaloTopology> topologyHandle;
  _setup.get<CaloTopologyRecord>().get(topologyHandle);
  topology_ = topologyHandle.product();
}

void
GEDPhotonGSCrysFixer::produce(edm::Event& _event, const edm::EventSetup& _setup)
{
  std::auto_ptr<reco::PhotonCollection> pOutput(new reco::PhotonCollection);
  
  auto&& inputPhotonsHandle(getHandle(_event, inputPhotonsToken_, "photons"));
  auto& inputPhotons(*inputPhotonsHandle);
  auto&& newCoresHandle(getHandle(_event, newCoresToken_, "newCores"));
  auto& newCoresToOldSCMap(*getHandle(_event, newCoresToOldSCMapToken_, "newCores"));
  auto& ebHits(*getHandle(_event, ebHitsToken_, "ebHits"));
  auto& vertices(*getHandle(_event, verticesToken_, "vertices"));

  energyCorrector_.init(_setup);
  if (energyCorrector_.gedRegression()) {
    energyCorrector_.gedRegression()->setEvent(_event);
    energyCorrector_.gedRegression()->setEventContent(_setup);
  }

  std::vector<reco::PhotonRef> oldPhotons;

  unsigned iP(0);
  for (auto& inPhoton : inputPhotons) {
    auto&& refToBase(inputPhotons.refAt(iP++));
    oldPhotons.emplace_back(refToBase.id(), &inPhoton, refToBase.key());

    auto&& newCore(getNewCore(inPhoton, newCoresHandle, newCoresToOldSCMap));

    auto& oldSC(*inPhoton.photonCore()->superCluster());
    auto& newSC(*newCore->superCluster());

    if (GainSwitchTools::hasEBGainSwitchIn5x5(oldSC, &ebHits, topology_)) {
      auto&& caloPosition(newSC.position());

      // will fix p4 later, but we apparently need the direction right
      auto&& direction((caloPosition - inPhoton.vertex()).unit());
      math::XYZTLorentzVectorD p4(direction.x(), direction.y(), direction.z(), 1.);

      pOutput->emplace_back(p4, caloPosition, newCore, inPhoton.vertex());
      auto& outPhoton(pOutput->back());
      
      reco::Photon::FiducialFlags fflags;
      fflags.isEB = inPhoton.isEB();
      fflags.isEE = inPhoton.isEE();
      fflags.isEBEtaGap = inPhoton.isEBEtaGap();
      fflags.isEBPhiGap = inPhoton.isEBPhiGap();
      fflags.isEERingGap = inPhoton.isEERingGap();
      fflags.isEEDeeGap = inPhoton.isEEDeeGap();
      fflags.isEBEEGap = inPhoton.isEBEEGap();

      outPhoton.setFiducialVolumeFlags(fflags);

      reco::Photon::IsolationVariables iso04;
      iso04.ecalRecHitSumEt = inPhoton.ecalRecHitSumEtConeDR04();
      iso04.hcalTowerSumEt = inPhoton.hcalTowerSumEtConeDR04();
      iso04.hcalDepth1TowerSumEt = inPhoton.hcalDepth1TowerSumEtConeDR04();
      iso04.hcalDepth2TowerSumEt = inPhoton.hcalDepth2TowerSumEtConeDR04();
      iso04.hcalTowerSumEtBc = inPhoton.hcalTowerSumEtBcConeDR04();
      iso04.hcalDepth1TowerSumEtBc = inPhoton.hcalDepth1TowerSumEtBcConeDR04();
      iso04.hcalDepth2TowerSumEtBc = inPhoton.hcalDepth2TowerSumEtBcConeDR04();
      iso04.trkSumPtSolidCone = inPhoton.trkSumPtSolidConeDR04();
      iso04.trkSumPtHollowCone = inPhoton.trkSumPtHollowConeDR04();
      iso04.nTrkSolidCone = inPhoton.nTrkSolidConeDR04();
      iso04.nTrkHollowCone = inPhoton.nTrkHollowConeDR04();

      reco::Photon::IsolationVariables iso03;
      iso03.ecalRecHitSumEt = inPhoton.ecalRecHitSumEtConeDR03();
      iso03.hcalTowerSumEt = inPhoton.hcalTowerSumEtConeDR03();
      iso03.hcalDepth1TowerSumEt = inPhoton.hcalDepth1TowerSumEtConeDR03();
      iso03.hcalDepth2TowerSumEt = inPhoton.hcalDepth2TowerSumEtConeDR03();
      iso03.hcalTowerSumEtBc = inPhoton.hcalTowerSumEtBcConeDR03();
      iso03.hcalDepth1TowerSumEtBc = inPhoton.hcalDepth1TowerSumEtBcConeDR03();
      iso03.hcalDepth2TowerSumEtBc = inPhoton.hcalDepth2TowerSumEtBcConeDR03();
      iso03.trkSumPtSolidCone = inPhoton.trkSumPtSolidConeDR03();
      iso03.trkSumPtHollowCone = inPhoton.trkSumPtHollowConeDR03();
      iso03.nTrkSolidCone = inPhoton.nTrkSolidConeDR03();
      iso03.nTrkHollowCone = inPhoton.nTrkHollowConeDR03();

      outPhoton.setIsolationVariables(iso04, iso03);

      auto& newSeed(*newSC.seed());

      auto&& cov(EcalClusterTools::covariances(newSeed, &ebHits, topology_, geometry_));
      auto&& locCov(EcalClusterTools::localCovariances(newSeed, &ebHits, topology_));
      auto&& cov55(noZS::EcalClusterTools::covariances(newSeed, &ebHits, topology_, geometry_));
      auto&& locCov55(noZS::EcalClusterTools::localCovariances(newSeed, &ebHits, topology_));

      auto& oldSS(inPhoton.showerShapeVariables());
      reco::Photon::ShowerShape newSS;

      newSS.e1x5 = EcalClusterTools::e1x5(newSeed, &ebHits, topology_);
      newSS.e2x5 = EcalClusterTools::e2x5Max(newSeed, &ebHits, topology_);
      newSS.e3x3 = EcalClusterTools::e3x3(newSeed, &ebHits, topology_);
      newSS.e5x5 = EcalClusterTools::e5x5(newSeed, &ebHits, topology_);
      newSS.maxEnergyXtal = EcalClusterTools::eMax(newSeed, &ebHits);
      newSS.sigmaEtaEta = std::sqrt(cov[0]);
      newSS.sigmaIetaIeta = std::sqrt(locCov[0]);
      newSS.hcalDepth1OverEcal = oldSS.hcalDepth1OverEcal * oldSC.energy() / newSC.energy();
      newSS.hcalDepth2OverEcal = oldSS.hcalDepth2OverEcal * oldSC.energy() / newSC.energy();
      newSS.hcalDepth1OverEcalBc = oldSS.hcalDepth1OverEcalBc * oldSC.energy() / newSC.energy();
      newSS.hcalDepth2OverEcalBc = oldSS.hcalDepth2OverEcalBc * oldSC.energy() / newSC.energy();
      newSS.hcalTowersBehindClusters = oldSS.hcalTowersBehindClusters;
      newSS.sigmaIetaIphi = locCov[1];
      newSS.sigmaIphiIphi = (edm::isFinite(locCov[2]) ? std::sqrt(locCov[2]) : 0.);
      newSS.e2nd = EcalClusterTools::e2nd(newSeed, &ebHits);
      newSS.eTop = EcalClusterTools::eTop(newSeed, &ebHits, topology_);
      newSS.eLeft = EcalClusterTools::eLeft(newSeed, &ebHits, topology_);
      newSS.eRight = EcalClusterTools::eRight(newSeed, &ebHits, topology_);
      newSS.eBottom = EcalClusterTools::eBottom(newSeed, &ebHits, topology_);
      newSS.e1x3 = EcalClusterTools::e1x3(newSeed, &ebHits, topology_);
      newSS.e2x2 = EcalClusterTools::e2x2(newSeed, &ebHits, topology_);
      newSS.e2x5Max = EcalClusterTools::e2x5Max(newSeed, &ebHits, topology_);
      newSS.e2x5Left = EcalClusterTools::e2x5Left(newSeed, &ebHits, topology_);
      newSS.e2x5Right = EcalClusterTools::e2x5Right(newSeed, &ebHits, topology_);
      newSS.e2x5Top = EcalClusterTools::e2x5Top(newSeed, &ebHits, topology_);
      newSS.e2x5Bottom = EcalClusterTools::e2x5Bottom(newSeed, &ebHits, topology_);
      newSS.effSigmaRR = oldSS.effSigmaRR;

      outPhoton.setShowerShapeVariables(newSS);

      reco::Photon::ShowerShape new55SS;
      new55SS.e1x5 = noZS::EcalClusterTools::e1x5(newSeed, &ebHits, topology_);
      new55SS.e2x5 = noZS::EcalClusterTools::e2x5Max(newSeed, &ebHits, topology_);
      new55SS.e3x3 = noZS::EcalClusterTools::e3x3(newSeed, &ebHits, topology_);
      new55SS.e5x5 = noZS::EcalClusterTools::e5x5(newSeed, &ebHits, topology_);
      new55SS.maxEnergyXtal = noZS::EcalClusterTools::eMax(newSeed, &ebHits);
      new55SS.sigmaEtaEta = std::sqrt(cov55[0]);
      new55SS.sigmaIetaIeta = std::sqrt(locCov55[0]);
      new55SS.sigmaIetaIphi = locCov55[1];
      new55SS.sigmaIphiIphi = (edm::isFinite(locCov55[2]) ? std::sqrt(locCov55[2]) : 0.);
      new55SS.e2nd = noZS::EcalClusterTools::e2nd(newSeed, &ebHits);
      new55SS.eTop = noZS::EcalClusterTools::eTop(newSeed, &ebHits, topology_);
      new55SS.eLeft = noZS::EcalClusterTools::eLeft(newSeed, &ebHits, topology_);
      new55SS.eRight = noZS::EcalClusterTools::eRight(newSeed, &ebHits, topology_);
      new55SS.eBottom = noZS::EcalClusterTools::eBottom(newSeed, &ebHits, topology_);
      new55SS.e1x3 = noZS::EcalClusterTools::e1x3(newSeed, &ebHits, topology_);
      new55SS.e2x2 = noZS::EcalClusterTools::e2x2(newSeed, &ebHits, topology_);
      new55SS.e2x5Max = noZS::EcalClusterTools::e2x5Max(newSeed, &ebHits, topology_);
      new55SS.e2x5Left = noZS::EcalClusterTools::e2x5Left(newSeed, &ebHits, topology_);
      new55SS.e2x5Right = noZS::EcalClusterTools::e2x5Right(newSeed, &ebHits, topology_);
      new55SS.e2x5Top = noZS::EcalClusterTools::e2x5Top(newSeed, &ebHits, topology_);
      new55SS.e2x5Bottom = noZS::EcalClusterTools::e2x5Bottom(newSeed, &ebHits, topology_);
      new55SS.effSigmaRR = oldSS.effSigmaRR;

      outPhoton.full5x5_setShowerShapeVariables(new55SS);

      reco::Photon::MIPVariables mipVars;
      mipVars.mipChi2 = inPhoton.mipChi2();
      mipVars.mipTotEnergy = inPhoton.mipTotEnergy();
      mipVars.mipSlope = inPhoton.mipSlope();
      mipVars.mipIntercept = inPhoton.mipIntercept();
      mipVars.mipNhitCone = inPhoton.mipNhitCone();
      mipVars.mipIsHalo = inPhoton.mipIsHalo();

      outPhoton.setMIPVariables(mipVars);

      outPhoton.setPflowIsolationVariables(inPhoton.getPflowIsolationVariables());

      reco::Photon::PflowIDVariables pfid;
      pfid.nClusterOutsideMustache = inPhoton.nClusterOutsideMustache();
      pfid.etOutsideMustache = inPhoton.etOutsideMustache();
      pfid.mva = inPhoton.pfMVA();
      
      outPhoton.setPflowIDVariables(pfid);

      energyCorrector_.calculate(_event, outPhoton, newSeed.hitsAndFractions()[0].first.subdetId(), vertices, _setup);

      outPhoton.setP4(outPhoton.p4(inPhoton.getCandidateP4type()));
      outPhoton.setCandidateP4type(inPhoton.getCandidateP4type());
    }
    else {
      pOutput->emplace_back(inPhoton);
      auto& outPhoton(pOutput->back());
      outPhoton.setPhotonCore(newCore);
    }
  }

  edm::OrphanHandle<reco::PhotonCollection> newPhotonsHandle(_event.put(pOutput));

  std::auto_ptr<PhotonRefMap> pRefMap(new PhotonRefMap);
  PhotonRefMap::Filler refMapFiller(*pRefMap);
  refMapFiller.insert(newPhotonsHandle, oldPhotons.begin(), oldPhotons.end());
  refMapFiller.fill();
  _event.put(pRefMap);
}
