//
// Class: PFSuperClusterTreeMaker.cc
//
// Info: Processes a track into histograms of delta-phis and such
//
// Author: L. Gray (FNAL)
//

#include <memory>
#include <map>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"
#include "TVector2.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"

#include "RecoEcal/EgammaCoreTools/interface/Mustache.h"
#include "CondFormats/EcalObjects/interface/EcalMustacheSCParameters.h"
#include "CondFormats/DataRecord/interface/EcalMustacheSCParametersRcd.h"
#include "CondFormats/EcalObjects/interface/EcalSCDynamicDPhiParameters.h"
#include "CondFormats/DataRecord/interface/EcalSCDynamicDPhiParametersRcd.h"
namespace MK = reco::MustacheKernel;

#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"

#include <algorithm>
#include <memory>

typedef edm::ParameterSet PSet;

namespace {
  template <typename T>
  void array_deleter(T* arr) {
    delete[] arr;
  }

  struct GetSharedRecHitFraction {
    const edm::Ptr<reco::CaloCluster> the_seed;
    double x_rechits_tot, x_rechits_match;
    GetSharedRecHitFraction(const edm::Ptr<reco::CaloCluster>& s) : the_seed(s) {}
    double operator()(const edm::Ptr<reco::CaloCluster>& x) {
      // now see if the clusters overlap in rechits
      const auto& seedHitsAndFractions = the_seed->hitsAndFractions();
      const auto& xHitsAndFractions = x->hitsAndFractions();
      x_rechits_tot = xHitsAndFractions.size();
      x_rechits_match = 0.0;
      for (const std::pair<DetId, float>& seedHit : seedHitsAndFractions) {
        for (const std::pair<DetId, float>& xHit : xHitsAndFractions) {
          if (seedHit.first == xHit.first) {
            x_rechits_match += 1.0;
          }
        }
      }
      return x_rechits_match / x_rechits_tot;
    }
  };
}  // namespace

class PFSuperClusterTreeMaker : public edm::EDAnalyzer {
  typedef TTree* treeptr;

public:
  PFSuperClusterTreeMaker(const PSet&);
  ~PFSuperClusterTreeMaker() {}

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  edm::Service<TFileService> _fs;
  bool _dogen;
  edm::InputTag _geninput;
  edm::InputTag _vtxsrc;
  bool _readEB, _readEE;
  edm::InputTag _scInputEB, _scInputEE;
  std::shared_ptr<PFEnergyCalibration> _calib;
  std::map<reco::SuperClusterRef, reco::GenParticleRef> _genmatched;
  void findBestGenMatches(const edm::Event& e, const edm::Handle<reco::SuperClusterCollection>& scs);
  void processSuperClusterFillTree(const edm::Event&, const reco::SuperClusterRef&);

  // SC parameters
  edm::ESGetToken<EcalMustacheSCParameters, EcalMustacheSCParametersRcd> ecalMustacheSCParametersToken_;
  const EcalMustacheSCParameters* mustacheSCParams_;
  edm::ESGetToken<EcalSCDynamicDPhiParameters, EcalSCDynamicDPhiParametersRcd> ecalSCDynamicDPhiParametersToken_;
  const EcalSCDynamicDPhiParameters* scDynamicDPhiParams_;

  // the tree
  void setTreeArraysForSize(const size_t N_ECAL, const size_t N_PS);
  treeptr _tree;
  Int_t nVtx;
  Float_t scRawEnergy, scCalibratedEnergy, scPreshowerEnergy, scEta, scPhi, scR, scPhiWidth, scEtaWidth,
      scSeedRawEnergy, scSeedCalibratedEnergy, scSeedEta, scSeedPhi;
  Float_t genEnergy, genEta, genPhi, genDRToCentroid, genDRToSeed;
  Int_t N_ECALClusters;
  std::shared_ptr<Float_t> clusterRawEnergy, clusterCalibEnergy, clusterEta, clusterPhi, clusterDPhiToSeed,
      clusterDEtaToSeed, clusterDPhiToCentroid, clusterDEtaToCentroid, clusterDPhiToGen, clusterDEtaToGen,
      clusterHitFractionSharedWithSeed;
  std::shared_ptr<Int_t> clusterInMustache, clusterInDynDPhi;
  Int_t N_PSClusters;
  std::shared_ptr<Float_t> psClusterRawEnergy, psClusterEta, psClusterPhi;
};

void PFSuperClusterTreeMaker::analyze(const edm::Event& e, const edm::EventSetup& es) {
  mustacheSCParams_ = &es.getData(ecalMustacheSCParametersToken_);
  scDynamicDPhiParams_ = &es.getData(ecalSCDynamicDPhiParametersToken_);

  edm::Handle<reco::VertexCollection> vtcs;
  e.getByLabel(_vtxsrc, vtcs);
  if (vtcs.isValid())
    nVtx = vtcs->size();
  else
    nVtx = -1;

  edm::Handle<reco::SuperClusterCollection> ebSCs, eeSCs;
  if (_readEB)
    e.getByLabel(_scInputEB, ebSCs);
  if (_readEE)
    e.getByLabel(_scInputEE, eeSCs);

  if (ebSCs.isValid()) {
    findBestGenMatches(e, ebSCs);
    for (size_t i = 0; i < ebSCs->size(); ++i) {
      processSuperClusterFillTree(e, reco::SuperClusterRef(ebSCs, i));
    }
  }

  if (eeSCs.isValid()) {
    findBestGenMatches(e, eeSCs);
    for (size_t i = 0; i < eeSCs->size(); ++i) {
      processSuperClusterFillTree(e, reco::SuperClusterRef(eeSCs, i));
    }
  }
}

void PFSuperClusterTreeMaker::findBestGenMatches(const edm::Event& e,
                                                 const edm::Handle<reco::SuperClusterCollection>& scs) {
  _genmatched.clear();
  reco::GenParticleRef genmatch;
  // gen information (if needed)
  if (_dogen) {
    edm::Handle<reco::GenParticleCollection> genp;
    std::vector<reco::GenParticleRef> elesandphos;
    e.getByLabel(_geninput, genp);
    if (genp.isValid()) {
      reco::GenParticleRef bestmatch;
      for (size_t i = 0; i < genp->size(); ++i) {
        const int pdgid = std::abs(genp->at(i).pdgId());
        if (pdgid == 22 || pdgid == 11) {
          elesandphos.push_back(reco::GenParticleRef(genp, i));
        }
      }
      for (size_t i = 0; i < elesandphos.size(); ++i) {
        double dE_min = -1;
        reco::SuperClusterRef bestmatch;
        for (size_t k = 0; k < scs->size(); ++k) {
          if (reco::deltaR(scs->at(k), *elesandphos[i]) < 0.3) {
            double dE = std::abs(scs->at(k).energy() - elesandphos[i]->energy());
            if (dE_min == -1 || dE < dE_min) {
              dE_min = dE;
              bestmatch = reco::SuperClusterRef(scs, k);
            }
          }
        }
        _genmatched[bestmatch] = elesandphos[i];
      }
    } else {
      throw cms::Exception("PFSuperClusterTreeMaker")
          << "Requested generator level information was not available!" << std::endl;
    }
  }
}

void PFSuperClusterTreeMaker::processSuperClusterFillTree(const edm::Event& e, const reco::SuperClusterRef& sc) {
  const int N_ECAL = sc->clustersSize();
  const int N_PS = sc->preshowerClustersSize();
  const double sc_eta = std::abs(sc->position().Eta());
  const double sc_cosheta = std::cosh(sc_eta);
  const double sc_pt = sc->rawEnergy() / sc_cosheta;
  if (!std::distance(sc->clustersBegin(), sc->clustersEnd()))
    return;
  if ((sc_pt < 3.0 && sc_eta < 2.0) || (sc_pt < 4.0 && sc_eta < 2.5 && sc_eta > 2.0) || (sc_pt < 6.0 && sc_eta > 2.5))
    return;
  N_ECALClusters = std::max(0, N_ECAL - 1);  // minus 1 because of seed
  N_PSClusters = N_PS;
  // gen information (if needed)
  reco::GenParticleRef genmatch;
  if (_dogen) {
    std::map<reco::SuperClusterRef, reco::GenParticleRef>::iterator itrmatch;
    if ((itrmatch = _genmatched.find(sc)) != _genmatched.end()) {
      genmatch = itrmatch->second;
      genEnergy = genmatch->energy();
      genEta = genmatch->eta();
      genPhi = genmatch->phi();
      genDRToCentroid = reco::deltaR(*sc, *genmatch);
      genDRToSeed = reco::deltaR(*genmatch, **(sc->clustersBegin()));
    } else {
      genEnergy = -1.0;
      genEta = 999.0;
      genPhi = 999.0;
      genDRToCentroid = 999.0;
      genDRToSeed = 999.0;
    }
  }
  // supercluster information
  setTreeArraysForSize(N_ECALClusters, N_PSClusters);
  scRawEnergy = sc->rawEnergy();
  scCalibratedEnergy = sc->energy();
  scPreshowerEnergy = sc->preshowerEnergy();
  scEta = sc->position().Eta();
  scPhi = sc->position().Phi();
  scR = sc->position().R();
  scPhiWidth = sc->phiWidth();
  scEtaWidth = sc->etaWidth();
  // sc seed information
  bool sc_is_pf = false;
  edm::Ptr<reco::CaloCluster> theseed = sc->seed();
  reco::CaloCluster_iterator startCluster = sc->clustersBegin();
  if (theseed.isNull() || !theseed.isAvailable()) {
    edm::Ptr<reco::CaloCluster> theseed = *(startCluster++);
  }
  edm::Ptr<reco::PFCluster> seedasPF = edm::Ptr<reco::PFCluster>(theseed);
  if (seedasPF.isNonnull() && seedasPF.isAvailable())
    sc_is_pf = true;
  GetSharedRecHitFraction fractionOfSeed(theseed);
  scSeedRawEnergy = theseed->energy();
  if (sc_is_pf) {
    scSeedCalibratedEnergy = _calib->energyEm(*seedasPF, 0.0, 0.0, false);
  } else {
    scSeedCalibratedEnergy = theseed->energy();
  }
  scSeedEta = theseed->eta();
  scSeedPhi = theseed->phi();
  // loop over all clusters that aren't the seed
  auto clusend = sc->clustersEnd();
  size_t iclus = 0;
  edm::Ptr<reco::PFCluster> pclus;
  for (auto clus = startCluster; clus != clusend; ++clus) {
    if (theseed == *clus)
      continue;
    clusterRawEnergy.get()[iclus] = (*clus)->energy();
    if (sc_is_pf) {
      pclus = edm::Ptr<reco::PFCluster>(*clus);
      clusterCalibEnergy.get()[iclus] = _calib->energyEm(*pclus, 0.0, 0.0, false);
    } else {
      clusterCalibEnergy.get()[iclus] = (*clus)->energy();
    }
    clusterEta.get()[iclus] = (*clus)->eta();
    clusterPhi.get()[iclus] = (*clus)->phi();
    clusterDPhiToSeed.get()[iclus] = TVector2::Phi_mpi_pi((*clus)->phi() - theseed->phi());
    clusterDEtaToSeed.get()[iclus] = (*clus)->eta() - theseed->eta();
    clusterDPhiToCentroid.get()[iclus] = TVector2::Phi_mpi_pi((*clus)->phi() - sc->phi());
    clusterDEtaToCentroid.get()[iclus] = (*clus)->eta() - sc->eta();
    clusterDPhiToCentroid.get()[iclus] = TVector2::Phi_mpi_pi((*clus)->phi() - sc->phi());
    clusterDEtaToCentroid.get()[iclus] = (*clus)->eta() - sc->eta();
    clusterHitFractionSharedWithSeed.get()[iclus] = fractionOfSeed(*clus);
    if (_dogen && genmatch.isNonnull()) {
      clusterDPhiToGen.get()[iclus] = TVector2::Phi_mpi_pi((*clus)->phi() - genmatch->phi());
      clusterDEtaToGen.get()[iclus] = (*clus)->eta() - genmatch->eta();
    }
    clusterInMustache.get()[iclus] = (Int_t)MK::inMustache(
        mustacheSCParams_, theseed->eta(), theseed->phi(), (*clus)->energy(), (*clus)->eta(), (*clus)->phi());

    clusterInDynDPhi.get()[iclus] = (Int_t)MK::inDynamicDPhiWindow(
        scDynamicDPhiParams_, theseed->eta(), theseed->phi(), (*clus)->energy(), (*clus)->eta(), (*clus)->phi());
    ++iclus;
  }
  // loop over all preshower clusters
  auto psclusend = sc->preshowerClustersEnd();
  size_t ipsclus = 0;
  edm::Ptr<reco::CaloCluster> ppsclus;
  for (auto psclus = sc->preshowerClustersBegin(); psclus != psclusend; ++psclus) {
    ppsclus = edm::Ptr<reco::CaloCluster>(*psclus);
    psClusterRawEnergy.get()[ipsclus] = ppsclus->energy();
    psClusterEta.get()[ipsclus] = ppsclus->eta();
    psClusterPhi.get()[ipsclus] = ppsclus->phi();
    ++ipsclus;
  }
  _tree->Fill();
}

PFSuperClusterTreeMaker::PFSuperClusterTreeMaker(const PSet& p) {
  ecalMustacheSCParametersToken_ = esConsumes<EcalMustacheSCParameters, EcalMustacheSCParametersRcd>();
  ecalSCDynamicDPhiParametersToken_ = esConsumes<EcalSCDynamicDPhiParameters, EcalSCDynamicDPhiParametersRcd>();

  _calib.reset(new PFEnergyCalibration());
  N_ECALClusters = 1;
  N_PSClusters = 1;
  _tree = _fs->make<TTree>("SuperClusterTree", "Dump of all available SC info");
  _tree->Branch("N_ECALClusters", &N_ECALClusters, "N_ECALClusters/I");
  _tree->Branch("N_PSClusters", &N_PSClusters, "N_PSClusters/I");
  _tree->Branch("nVtx", &nVtx, "nVtx/I");
  _tree->Branch("scRawEnergy", &scRawEnergy, "scRawEnergy/F");
  _tree->Branch("scCalibratedEnergy", &scCalibratedEnergy, "scCalibratedEnergy/F");
  _tree->Branch("scPreshowerEnergy", &scPreshowerEnergy, "scPreshowerEnergy/F");
  _tree->Branch("scEta", &scEta, "scEta/F");
  _tree->Branch("scPhi", &scPhi, "scPhi/F");
  _tree->Branch("scR", &scR, "scR/F");
  _tree->Branch("scPhiWidth", &scPhiWidth, "scPhiWidth/F");
  _tree->Branch("scEtaWidth", &scEtaWidth, "scEtaWidth/F");
  _tree->Branch("scSeedRawEnergy", &scSeedRawEnergy, "scSeedRawEnergy/F");
  _tree->Branch("scSeedCalibratedEnergy", &scSeedCalibratedEnergy, "scSeedCalibratedEnergy/F");
  _tree->Branch("scSeedEta", &scSeedEta, "scSeedEta/F");
  _tree->Branch("scSeedPhi", &scSeedPhi, "scSeedPhi/F");
  // ecal cluster information
  clusterRawEnergy.reset(new Float_t[1], array_deleter<Float_t>);
  _tree->Branch("clusterRawEnergy", clusterRawEnergy.get(), "clusterRawEnergy[N_ECALClusters]/F");
  clusterCalibEnergy.reset(new Float_t[1], array_deleter<Float_t>);
  _tree->Branch("clusterCalibEnergy", clusterCalibEnergy.get(), "clusterCalibEnergy[N_ECALClusters]/F");
  clusterEta.reset(new Float_t[1], array_deleter<Float_t>);
  _tree->Branch("clusterEta", clusterEta.get(), "clusterEta[N_ECALClusters]/F");
  clusterPhi.reset(new Float_t[1], array_deleter<Float_t>);
  _tree->Branch("clusterPhi", clusterPhi.get(), "clusterPhi[N_ECALClusters]/F");
  clusterDPhiToSeed.reset(new Float_t[1], array_deleter<Float_t>);
  _tree->Branch("clusterDPhiToSeed", clusterDPhiToSeed.get(), "clusterDPhiToSeed[N_ECALClusters]/F");
  clusterDEtaToSeed.reset(new Float_t[1], array_deleter<Float_t>);
  _tree->Branch("clusterDEtaToSeed", clusterDEtaToSeed.get(), "clusterDEtaToSeed[N_ECALClusters]/F");
  clusterDPhiToCentroid.reset(new Float_t[1], array_deleter<Float_t>);
  _tree->Branch("clusterDPhiToCentroid", clusterDPhiToCentroid.get(), "clusterDPhiToCentroid[N_ECALClusters]/F");
  clusterDEtaToCentroid.reset(new Float_t[1], array_deleter<Float_t>);
  _tree->Branch("clusterDEtaToCentroid", clusterDEtaToCentroid.get(), "clusterDEtaToCentroid[N_ECALClusters]/F");
  clusterHitFractionSharedWithSeed.reset(new Float_t[1], array_deleter<Float_t>);
  _tree->Branch("clusterHitFractionSharedWithSeed",
                clusterHitFractionSharedWithSeed.get(),
                "clusterHitFractionSharedWithSeed[N_ECALClusters]/F");
  clusterInMustache.reset(new Int_t[1], array_deleter<Int_t>);
  _tree->Branch("clusterInMustache", clusterInMustache.get(), "clusterInMustache[N_ECALClusters]/I");
  clusterInDynDPhi.reset(new Int_t[1], array_deleter<Int_t>);
  _tree->Branch("clusterInDynDPhi", clusterInDynDPhi.get(), "clusterInDynDPhi[N_ECALClusters]/I");
  // preshower information
  psClusterRawEnergy.reset(new Float_t[1], array_deleter<Float_t>);
  _tree->Branch("psClusterRawEnergy", psClusterRawEnergy.get(), "psClusterRawEnergy[N_PSClusters]/F");
  psClusterEta.reset(new Float_t[1], array_deleter<Float_t>);
  _tree->Branch("psClusterEta", psClusterEta.get(), "psClusterEta[N_PSClusters]/F");
  psClusterPhi.reset(new Float_t[1], array_deleter<Float_t>);
  _tree->Branch("psClusterPhi", psClusterPhi.get(), "psClusterPhi[N_PSClusters]/F");

  if ((_dogen = p.getUntrackedParameter<bool>("doGen", false))) {
    _geninput = p.getParameter<edm::InputTag>("genSrc");
    _tree->Branch("genEta", &genEta, "genEta/F");
    _tree->Branch("genPhi", &genPhi, "genPhi/F");
    _tree->Branch("genEnergy", &genEnergy, "genEnergy/F");
    _tree->Branch("genDRToCentroid", &genDRToCentroid, "genDRToCentroid/F");
    _tree->Branch("genDRToSeed", &genDRToSeed, "genDRToSeed/F");

    clusterDPhiToGen.reset(new Float_t[1], array_deleter<Float_t>);
    _tree->Branch("clusterDPhiToGen", clusterDPhiToGen.get(), "clusterDPhiToGen[N_ECALClusters]/F");
    clusterDEtaToGen.reset(new Float_t[1], array_deleter<Float_t>);
    _tree->Branch("clusterDEtaToGen", clusterDEtaToGen.get(), "clusterDPhiToGen[N_ECALClusters]/F");
  }
  _vtxsrc = p.getParameter<edm::InputTag>("primaryVertices");

  _readEB = _readEE = false;
  if (p.existsAs<edm::InputTag>("superClusterSrcEB")) {
    _readEB = true;
    _scInputEB = p.getParameter<edm::InputTag>("superClusterSrcEB");
  }
  if (p.existsAs<edm::InputTag>("superClusterSrcEE")) {
    _readEE = true;
    _scInputEE = p.getParameter<edm::InputTag>("superClusterSrcEE");
  }
}

void PFSuperClusterTreeMaker::setTreeArraysForSize(const size_t N_ECAL, const size_t N_PS) {
  Float_t* cRE_new = new Float_t[N_ECAL];
  clusterRawEnergy.reset(cRE_new, array_deleter<Float_t>);
  _tree->GetBranch("clusterRawEnergy")->SetAddress(clusterRawEnergy.get());
  Float_t* cCE_new = new Float_t[N_ECAL];
  clusterCalibEnergy.reset(cCE_new, array_deleter<Float_t>);
  _tree->GetBranch("clusterCalibEnergy")->SetAddress(clusterCalibEnergy.get());
  Float_t* cEta_new = new Float_t[N_ECAL];
  clusterEta.reset(cEta_new, array_deleter<Float_t>);
  _tree->GetBranch("clusterEta")->SetAddress(clusterEta.get());
  Float_t* cPhi_new = new Float_t[N_ECAL];
  clusterPhi.reset(cPhi_new, array_deleter<Float_t>);
  _tree->GetBranch("clusterPhi")->SetAddress(clusterPhi.get());
  Float_t* cDPhiSeed_new = new Float_t[N_ECAL];
  clusterDPhiToSeed.reset(cDPhiSeed_new, array_deleter<Float_t>);
  _tree->GetBranch("clusterDPhiToSeed")->SetAddress(clusterDPhiToSeed.get());
  Float_t* cDEtaSeed_new = new Float_t[N_ECAL];
  clusterDEtaToSeed.reset(cDEtaSeed_new, array_deleter<Float_t>);
  _tree->GetBranch("clusterDEtaToSeed")->SetAddress(clusterDEtaToSeed.get());
  Float_t* cDPhiCntr_new = new Float_t[N_ECAL];
  clusterDPhiToCentroid.reset(cDPhiCntr_new, array_deleter<Float_t>);
  _tree->GetBranch("clusterDPhiToCentroid")->SetAddress(clusterDPhiToCentroid.get());
  Float_t* cDEtaCntr_new = new Float_t[N_ECAL];
  clusterDEtaToCentroid.reset(cDEtaCntr_new, array_deleter<Float_t>);
  _tree->GetBranch("clusterDEtaToCentroid")->SetAddress(clusterDEtaToCentroid.get());
  Float_t* cHitFracShared_new = new Float_t[N_ECAL];
  clusterHitFractionSharedWithSeed.reset(cHitFracShared_new, array_deleter<Float_t>);
  _tree->GetBranch("clusterHitFractionSharedWithSeed")->SetAddress(clusterHitFractionSharedWithSeed.get());

  if (_dogen) {
    Float_t* cDPhiGen_new = new Float_t[N_ECAL];
    clusterDPhiToGen.reset(cDPhiGen_new, array_deleter<Float_t>);
    _tree->GetBranch("clusterDPhiToGen")->SetAddress(clusterDPhiToGen.get());
    Float_t* cDEtaGen_new = new Float_t[N_ECAL];
    clusterDEtaToGen.reset(cDEtaGen_new, array_deleter<Float_t>);
    _tree->GetBranch("clusterDEtaToGen")->SetAddress(clusterDEtaToGen.get());
  }
  Int_t* cInMust_new = new Int_t[N_ECAL];
  clusterInMustache.reset(cInMust_new, array_deleter<Int_t>);
  _tree->GetBranch("clusterInMustache")->SetAddress(clusterInMustache.get());
  Int_t* cInDynDPhi_new = new Int_t[N_ECAL];
  clusterInDynDPhi.reset(cInDynDPhi_new, array_deleter<Int_t>);
  _tree->GetBranch("clusterInDynDPhi")->SetAddress(clusterInDynDPhi.get());
  Float_t* psRE_new = new Float_t[N_PS];
  psClusterRawEnergy.reset(psRE_new, array_deleter<Float_t>);
  _tree->GetBranch("psClusterRawEnergy")->SetAddress(psClusterRawEnergy.get());
  Float_t* psEta_new = new Float_t[N_PS];
  psClusterEta.reset(psEta_new, array_deleter<Float_t>);
  _tree->GetBranch("psClusterEta")->SetAddress(psClusterEta.get());
  Float_t* psPhi_new = new Float_t[N_PS];
  psClusterPhi.reset(psPhi_new, array_deleter<Float_t>);
  _tree->GetBranch("psClusterPhi")->SetAddress(psClusterPhi.get());
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFSuperClusterTreeMaker);
