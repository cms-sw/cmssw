//
// Class: PFEGCandidateTreeMaker.cc
//
// Info: Outputs a tree with PF-EGamma information, mostly SC info.
//       Checks to see if the input EG candidates are matched to
//       some existing PF reco (PF-Photons and PF-Electrons).
//
// Author: L. Gray (FNAL)
//

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidatePhotonExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtra.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "RecoEcal/EgammaCoreTools/interface/Mustache.h"
#include "CondFormats/EcalObjects/interface/EcalMustacheSCParameters.h"
#include "CondFormats/DataRecord/interface/EcalMustacheSCParametersRcd.h"
#include "CondFormats/EcalObjects/interface/EcalSCDynamicDPhiParameters.h"
#include "CondFormats/DataRecord/interface/EcalSCDynamicDPhiParametersRcd.h"
#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"

#include "TTree.h"
#include "TVector2.h"

#include <algorithm>
#include <map>
#include <memory>

typedef edm::ParameterSet PSet;

namespace {
  template <typename T>
  struct array_deleter {
    void operator()(T* arr) { delete[] arr; }
  };

  struct GetSharedRecHitFraction {
    const edm::Ptr<reco::PFCluster> the_seed;
    double x_rechits_tot, x_rechits_match;
    GetSharedRecHitFraction(const edm::Ptr<reco::PFCluster>& s) : the_seed(s) {}
    double operator()(const edm::Ptr<reco::PFCluster>& x) {
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

class PFEGCandidateTreeMaker : public edm::EDAnalyzer {
  typedef TTree* treeptr;

public:
  PFEGCandidateTreeMaker(const PSet&);
  ~PFEGCandidateTreeMaker() {}

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  edm::Service<TFileService> _fs;
  bool _dogen;
  edm::InputTag _geninput;
  edm::InputTag _vtxsrc;
  edm::InputTag _pfEGInput;
  edm::InputTag _pfInput;
  PFEnergyCalibration _calib;
  std::map<reco::PFCandidateRef, reco::GenParticleRef> _genmatched;
  void findBestGenMatches(const edm::Event& e, const edm::Handle<reco::PFCandidateCollection>&);
  void processEGCandidateFillTree(const edm::Event&,
                                  const reco::PFCandidateRef&,
                                  const edm::Handle<reco::PFCandidateCollection>&);
  bool getPFCandMatch(const reco::PFCandidate&, const edm::Handle<reco::PFCandidateCollection>&, const int);

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
  Int_t hasParentSC, pfPhotonMatch, pfElectronMatch;
  Float_t genEnergy, genEta, genPhi, genDRToCentroid, genDRToSeed;
  Int_t N_ECALClusters;
  std::vector<Float_t> clusterRawEnergy;
  std::vector<Float_t> clusterCalibEnergy;
  std::vector<Float_t> clusterEta, clusterPhi, clusterDPhiToSeed, clusterDEtaToSeed, clusterDPhiToCentroid,
      clusterDEtaToCentroid, clusterDPhiToGen, clusterDEtaToGen, clusterHitFractionSharedWithSeed;
  std::vector<Int_t> clusterInMustache, clusterInDynDPhi;
  Int_t N_PSClusters;
  std::vector<Float_t> psClusterRawEnergy, psClusterEta, psClusterPhi;
};

void PFEGCandidateTreeMaker::analyze(const edm::Event& e, const edm::EventSetup& es) {
  mustacheSCParams_ = &es.getData(ecalMustacheSCParametersToken_);
  scDynamicDPhiParams_ = &es.getData(ecalSCDynamicDPhiParametersToken_);

  edm::Handle<reco::VertexCollection> vtcs;
  e.getByLabel(_vtxsrc, vtcs);
  if (vtcs.isValid())
    nVtx = vtcs->size();
  else
    nVtx = -1;

  edm::Handle<reco::PFCandidateCollection> pfEG;
  edm::Handle<reco::PFCandidateCollection> pfCands;
  e.getByLabel(_pfEGInput, pfEG);
  e.getByLabel(_pfInput, pfCands);

  if (pfEG.isValid()) {
    findBestGenMatches(e, pfEG);
    for (size_t i = 0; i < pfEG->size(); ++i) {
      processEGCandidateFillTree(e, reco::PFCandidateRef(pfEG, i), pfCands);
    }
  } else {
    throw cms::Exception("PFEGCandidateTreeMaker")
        << "Product ID for the EB SuperCluster collection was invalid!" << std::endl;
  }
}

void PFEGCandidateTreeMaker::findBestGenMatches(const edm::Event& e,
                                                const edm::Handle<reco::PFCandidateCollection>& pfs) {
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
        reco::PFCandidateRef bestmatch;
        for (size_t k = 0; k < pfs->size(); ++k) {
          reco::SuperClusterRef scref = pfs->at(k).superClusterRef();
          if (scref.isAvailable() && scref.isNonnull() && reco::deltaR(*scref, *elesandphos[i]) < 0.3) {
            double dE = std::abs(scref->energy() - elesandphos[i]->energy());
            if (dE_min == -1 || dE < dE_min) {
              dE_min = dE;
              bestmatch = reco::PFCandidateRef(pfs, k);
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

void PFEGCandidateTreeMaker::processEGCandidateFillTree(const edm::Event& e,
                                                        const reco::PFCandidateRef& pf,
                                                        const edm::Handle<reco::PFCandidateCollection>& pfCands) {
  if (pf->superClusterRef().isNull() || !pf->superClusterRef().isAvailable()) {
    return;
  }
  if (pf->egammaExtraRef().isNull() || !pf->egammaExtraRef().isAvailable()) {
    return;
  }
  const reco::SuperCluster& sc = *(pf->superClusterRef());
  reco::SuperClusterRef egsc = pf->egammaExtraRef()->superClusterPFECALRef();
  bool eleMatch(getPFCandMatch(*pf, pfCands, 11));
  bool phoMatch(getPFCandMatch(*pf, pfCands, 22));

  const int N_ECAL = sc.clustersSize();
  const int N_PS = sc.preshowerClustersSize();
  const double sc_eta = std::abs(sc.position().Eta());
  const double sc_cosheta = std::cosh(sc_eta);
  const double sc_pt = sc.rawEnergy() / sc_cosheta;
  if (!N_ECAL)
    return;
  if ((sc_pt < 3.0 && sc_eta < 2.0) || (sc_pt < 4.0 && sc_eta < 2.5 && sc_eta > 2.0) || (sc_pt < 6.0 && sc_eta > 2.5))
    return;
  N_ECALClusters = std::max(0, N_ECAL - 1);  // minus 1 because of seed
  N_PSClusters = N_PS;
  reco::GenParticleRef genmatch;
  // gen information (if needed)
  if (_dogen) {
    std::map<reco::PFCandidateRef, reco::GenParticleRef>::iterator itrmatch;
    if ((itrmatch = _genmatched.find(pf)) != _genmatched.end()) {
      genmatch = itrmatch->second;
      genEnergy = genmatch->energy();
      genEta = genmatch->eta();
      genPhi = genmatch->phi();
      genDRToCentroid = reco::deltaR(sc, *genmatch);
      genDRToSeed = reco::deltaR(*genmatch, **(sc.clustersBegin()));
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
  hasParentSC = (Int_t)(egsc.isAvailable() && egsc.isNonnull());
  pfElectronMatch = (Int_t)eleMatch;
  pfPhotonMatch = (Int_t)phoMatch;
  scRawEnergy = sc.rawEnergy();
  scCalibratedEnergy = sc.energy();
  scPreshowerEnergy = sc.preshowerEnergy();
  scEta = sc.position().Eta();
  scPhi = sc.position().Phi();
  scR = sc.position().R();
  scPhiWidth = sc.phiWidth();
  scEtaWidth = sc.etaWidth();
  // sc seed information
  edm::Ptr<reco::PFCluster> theseed = edm::Ptr<reco::PFCluster>(sc.seed());
  GetSharedRecHitFraction fractionOfSeed(theseed);
  scSeedRawEnergy = theseed->energy();
  scSeedCalibratedEnergy = _calib.energyEm(*theseed, 0.0, 0.0, false);
  scSeedEta = theseed->eta();
  scSeedPhi = theseed->phi();
  // loop over all clusters that aren't the seed
  auto clusend = sc.clustersEnd();
  size_t iclus = 0;
  edm::Ptr<reco::PFCluster> pclus;
  for (auto clus = sc.clustersBegin(); clus != clusend; ++clus) {
    pclus = edm::Ptr<reco::PFCluster>(*clus);
    if (theseed == pclus)
      continue;
    clusterRawEnergy[iclus] = pclus->energy();
    clusterCalibEnergy[iclus] = _calib.energyEm(*pclus, 0.0, 0.0, false);
    clusterEta[iclus] = pclus->eta();
    clusterPhi[iclus] = pclus->phi();
    clusterDPhiToSeed[iclus] = TVector2::Phi_mpi_pi(pclus->phi() - theseed->phi());
    clusterDEtaToSeed[iclus] = pclus->eta() - theseed->eta();
    clusterDPhiToCentroid[iclus] = TVector2::Phi_mpi_pi(pclus->phi() - sc.phi());
    clusterDEtaToCentroid[iclus] = pclus->eta() - sc.eta();
    clusterDPhiToCentroid[iclus] = TVector2::Phi_mpi_pi(pclus->phi() - sc.phi());
    clusterDEtaToCentroid[iclus] = pclus->eta() - sc.eta();
    clusterHitFractionSharedWithSeed[iclus] = fractionOfSeed(pclus);
    if (_dogen && genmatch.isNonnull()) {
      clusterDPhiToGen[iclus] = TVector2::Phi_mpi_pi(pclus->phi() - genmatch->phi());
      clusterDEtaToGen[iclus] = pclus->eta() - genmatch->eta();
    }
    clusterInMustache[iclus] = (Int_t)reco::MustacheKernel::inMustache(
        mustacheSCParams_, theseed->eta(), theseed->phi(), pclus->energy(), pclus->eta(), pclus->phi());
    clusterInDynDPhi[iclus] = (Int_t)reco::MustacheKernel::inDynamicDPhiWindow(scDynamicDPhiParams_,
                                                                               PFLayer::ECAL_BARREL == pclus->layer(),
                                                                               theseed->phi(),
                                                                               pclus->energy(),
                                                                               pclus->eta(),
                                                                               pclus->phi());
    ++iclus;
  }
  // loop over all preshower clusters
  auto psclusend = sc.preshowerClustersEnd();
  size_t ipsclus = 0;
  edm::Ptr<reco::PFCluster> ppsclus;
  for (auto psclus = sc.preshowerClustersBegin(); psclus != psclusend; ++psclus) {
    ppsclus = edm::Ptr<reco::PFCluster>(*psclus);
    psClusterRawEnergy[ipsclus] = ppsclus->energy();
    psClusterEta[ipsclus] = ppsclus->eta();
    psClusterPhi[ipsclus] = ppsclus->phi();
    ++ipsclus;
  }
  _tree->Fill();
}

bool PFEGCandidateTreeMaker::getPFCandMatch(const reco::PFCandidate& cand,
                                            const edm::Handle<reco::PFCandidateCollection>& pf,
                                            const int pdgid_search) {
  reco::PFCandidateEGammaExtraRef egxtra = cand.egammaExtraRef();
  if (egxtra.isAvailable() && egxtra.isNonnull()) {
    reco::SuperClusterRef scref = egxtra->superClusterPFECALRef();
    if (scref.isAvailable() && scref.isNonnull()) {
      for (auto ipf = pf->begin(); ipf != pf->end(); ++ipf) {
        if (std::abs(ipf->pdgId()) == pdgid_search && pdgid_search == 11) {
          reco::GsfTrackRef gsfref = ipf->gsfTrackRef();
          reco::ElectronSeedRef sRef = gsfref->seedRef().castTo<reco::ElectronSeedRef>();
          if (sRef.isNonnull() && sRef.isAvailable() && sRef->isEcalDriven()) {
            reco::SuperClusterRef temp(sRef->caloCluster().castTo<reco::SuperClusterRef>());
            if (scref == temp) {
              return true;
            }
          }
        } else if (std::abs(ipf->pdgId()) == 22 && pdgid_search == 22) {
          reco::SuperClusterRef temp(ipf->superClusterRef());
          if (scref == temp) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

PFEGCandidateTreeMaker::PFEGCandidateTreeMaker(const PSet& p) {
  ecalMustacheSCParametersToken_ = esConsumes<EcalMustacheSCParameters, EcalMustacheSCParametersRcd>();
  ecalSCDynamicDPhiParametersToken_ = esConsumes<EcalSCDynamicDPhiParameters, EcalSCDynamicDPhiParametersRcd>();

  N_ECALClusters = 1;
  N_PSClusters = 1;
  _tree = _fs->make<TTree>("SuperClusterTree", "Dump of all available SC info");
  _tree->Branch("N_ECALClusters", &N_ECALClusters, "N_ECALClusters/I");
  _tree->Branch("N_PSClusters", &N_PSClusters, "N_PSClusters/I");
  _tree->Branch("nVtx", &nVtx, "nVtx/I");
  _tree->Branch("hasParentSC", &hasParentSC, "hasParentSC/I");
  _tree->Branch("pfPhotonMatch", &pfPhotonMatch, "pfPhotonMatch/I");
  _tree->Branch("pfElectronMatch", &pfElectronMatch, "pfElectronMatch/I");
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
  clusterRawEnergy.resize(1);
  _tree->Branch("clusterRawEnergy", &clusterRawEnergy[0], "clusterRawEnergy[N_ECALClusters]/F");
  clusterCalibEnergy.resize(1);
  _tree->Branch("clusterCalibEnergy", &clusterCalibEnergy[0], "clusterCalibEnergy[N_ECALClusters]/F");
  clusterEta.resize(1);
  _tree->Branch("clusterEta", &clusterEta[0], "clusterEta[N_ECALClusters]/F");
  clusterPhi.resize(1);
  _tree->Branch("clusterPhi", &clusterPhi[0], "clusterPhi[N_ECALClusters]/F");
  clusterDPhiToSeed.resize(1);
  _tree->Branch("clusterDPhiToSeed", &clusterDPhiToSeed[0], "clusterDPhiToSeed[N_ECALClusters]/F");
  clusterDEtaToSeed.resize(1);
  _tree->Branch("clusterDEtaToSeed", &clusterDEtaToSeed[0], "clusterDEtaToSeed[N_ECALClusters]/F");
  clusterDPhiToCentroid.resize(1);
  _tree->Branch("clusterDPhiToCentroid", &clusterDPhiToCentroid[0], "clusterDPhiToCentroid[N_ECALClusters]/F");
  clusterDEtaToCentroid.resize(1);
  _tree->Branch("clusterDEtaToCentroid", &clusterDEtaToCentroid[0], "clusterDEtaToCentroid[N_ECALClusters]/F");
  clusterHitFractionSharedWithSeed.resize(1);
  _tree->Branch("clusterHitFractionSharedWithSeed",
                &clusterHitFractionSharedWithSeed[0],
                "clusterHitFractionSharedWithSeed[N_ECALClusters]/F");
  clusterInMustache.resize(1);
  _tree->Branch("clusterInMustache", &clusterInMustache[0], "clusterInMustache[N_ECALClusters]/I");
  clusterInDynDPhi.resize(1);
  _tree->Branch("clusterInDynDPhi", &clusterInDynDPhi[0], "clusterInDynDPhi[N_ECALClusters]/I");
  // preshower information
  psClusterRawEnergy.resize(1);
  _tree->Branch("psClusterRawEnergy", &psClusterRawEnergy[0], "psClusterRawEnergy[N_PSClusters]/F");
  psClusterEta.resize(1);
  _tree->Branch("psClusterEta", &psClusterEta[0], "psClusterEta[N_PSClusters]/F");
  psClusterPhi.resize(1);
  _tree->Branch("psClusterPhi", &psClusterPhi[0], "psClusterPhi[N_PSClusters]/F");

  if ((_dogen = p.getUntrackedParameter<bool>("doGen", false))) {
    _geninput = p.getParameter<edm::InputTag>("genSrc");
    _tree->Branch("genEta", &genEta, "genEta/F");
    _tree->Branch("genPhi", &genPhi, "genPhi/F");
    _tree->Branch("genEnergy", &genEnergy, "genEnergy/F");
    _tree->Branch("genDRToCentroid", &genDRToCentroid, "genDRToCentroid/F");
    _tree->Branch("genDRToSeed", &genDRToSeed, "genDRToSeed/F");

    clusterDPhiToGen.resize(1);
    _tree->Branch("clusterDPhiToGen", &clusterDPhiToGen[0], "clusterDPhiToGen[N_ECALClusters]/F");
    clusterDEtaToGen.resize(1);
    _tree->Branch("clusterDEtaToGen", &clusterDEtaToGen[0], "clusterDPhiToGen[N_ECALClusters]/F");
  }
  _vtxsrc = p.getParameter<edm::InputTag>("primaryVertices");
  _pfEGInput = p.getParameter<edm::InputTag>("pfEGammaCandSrc");
  _pfInput = p.getParameter<edm::InputTag>("pfCandSrc");
}

void PFEGCandidateTreeMaker::setTreeArraysForSize(const size_t N_ECAL, const size_t N_PS) {
  clusterRawEnergy.resize(N_ECAL);
  _tree->GetBranch("clusterRawEnergy")->SetAddress(&clusterRawEnergy[0]);
  clusterCalibEnergy.resize(N_ECAL);
  _tree->GetBranch("clusterCalibEnergy")->SetAddress(&clusterCalibEnergy[0]);
  clusterEta.resize(N_ECAL);
  _tree->GetBranch("clusterEta")->SetAddress(&clusterEta[0]);
  clusterPhi.resize(N_ECAL);
  _tree->GetBranch("clusterPhi")->SetAddress(&clusterPhi[0]);
  clusterDPhiToSeed.resize(N_ECAL);
  _tree->GetBranch("clusterDPhiToSeed")->SetAddress(&clusterDPhiToSeed[0]);
  clusterDEtaToSeed.resize(N_ECAL);
  _tree->GetBranch("clusterDEtaToSeed")->SetAddress(&clusterDEtaToSeed[0]);
  clusterDPhiToCentroid.resize(N_ECAL);
  _tree->GetBranch("clusterDPhiToCentroid")->SetAddress(&clusterDPhiToCentroid[0]);
  clusterDEtaToCentroid.resize(N_ECAL);
  _tree->GetBranch("clusterDEtaToCentroid")->SetAddress(&clusterDEtaToCentroid[0]);
  clusterHitFractionSharedWithSeed.resize(N_ECAL);
  _tree->GetBranch("clusterHitFractionSharedWithSeed")->SetAddress(&clusterHitFractionSharedWithSeed[0]);

  if (_dogen) {
    clusterDPhiToGen.resize(N_ECAL);
    _tree->GetBranch("clusterDPhiToGen")->SetAddress(&clusterDPhiToGen[0]);
    clusterDEtaToGen.resize(N_ECAL);
    _tree->GetBranch("clusterDEtaToGen")->SetAddress(&clusterDEtaToGen[0]);
  }
  clusterInMustache.resize(N_ECAL);
  _tree->GetBranch("clusterInMustache")->SetAddress(&clusterInMustache[0]);
  clusterInDynDPhi.resize(N_ECAL);
  _tree->GetBranch("clusterInDynDPhi")->SetAddress(&clusterInDynDPhi[0]);
  psClusterRawEnergy.resize(N_ECAL);
  _tree->GetBranch("psClusterRawEnergy")->SetAddress(&psClusterRawEnergy[0]);
  psClusterEta.resize(N_ECAL);
  _tree->GetBranch("psClusterEta")->SetAddress(&psClusterEta[0]);
  psClusterPhi.resize(N_ECAL);
  _tree->GetBranch("psClusterPhi")->SetAddress(&psClusterPhi[0]);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFEGCandidateTreeMaker);
