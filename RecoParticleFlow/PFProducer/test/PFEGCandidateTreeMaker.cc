//
// Class: PFEGCandidateTreeMaker.cc
//
// Info: Outputs a tree with PF-EGamma information, mostly SC info.
//       Checks to see if the input EG candidates are matched to 
//       some existing PF reco (PF-Photons and PF-Electrons).
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

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidatePhotonExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidatePhotonExtraFwd.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtra.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateEGammaExtraFwd.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"

#include "DataFormats/GsfTrackReco/interface/GsfTrackExtra.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackExtraFwd.h"

#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"

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
namespace MK = reco::MustacheKernel;

#include "RecoParticleFlow/PFClusterTools/interface/PFEnergyCalibration.h"

#include <algorithm>
#include <memory>

typedef edm::ParameterSet PSet;

namespace {  
  template<typename T>
  struct array_deleter{
    void operator () (T* arr) { delete [] arr; }
  };

  typedef std::unary_function<const edm::Ptr<reco::PFCluster>&, 
			      double> ClusUnaryFunction;  

  struct GetSharedRecHitFraction : public ClusUnaryFunction {
    const edm::Ptr<reco::PFCluster> the_seed;    
    double x_rechits_tot, x_rechits_match;
    GetSharedRecHitFraction(const edm::Ptr<reco::PFCluster>& s) : 
      the_seed(s) {}
    double operator()(const edm::Ptr<reco::PFCluster>& x) {      
      // now see if the clusters overlap in rechits
      const auto& seedHitsAndFractions = 
	the_seed->hitsAndFractions();
      const auto& xHitsAndFractions = 
	x->hitsAndFractions();      
      x_rechits_tot   = xHitsAndFractions.size();
      x_rechits_match = 0.0;      
      for( const std::pair<DetId, float>& seedHit : seedHitsAndFractions ) {
	for( const std::pair<DetId, float>& xHit : xHitsAndFractions ) {
	  if( seedHit.first == xHit.first ) {	    
	    x_rechits_match += 1.0;
	  }
	}	
      }      
      return x_rechits_match/x_rechits_tot;
    }
  };
}

class PFEGCandidateTreeMaker : public edm::EDAnalyzer {
  typedef TTree* treeptr;  
public:
  PFEGCandidateTreeMaker(const PSet&);
  ~PFEGCandidateTreeMaker() {}

  void analyze(const edm::Event&, const edm::EventSetup&);
private:    
  edm::Service<TFileService> _fs;
  bool _dogen;
  edm::InputTag _geninput;
  edm::InputTag _vtxsrc;
  edm::InputTag _pfEGInput;
  edm::InputTag _pfInput;
  std::shared_ptr<PFEnergyCalibration> _calib;
  std::map<reco::PFCandidateRef,reco::GenParticleRef> _genmatched;
  void findBestGenMatches(const edm::Event& e, 
			  const edm::Handle<reco::PFCandidateCollection>&);
  void processEGCandidateFillTree(const edm::Event&,
			     const reco::PFCandidateRef&,
			     const edm::Handle<reco::PFCandidateCollection>&);
  bool getPFCandMatch(const reco::PFCandidate&,
		      const edm::Handle<reco::PFCandidateCollection>&,
		      const int );
  // the tree  
  void setTreeArraysForSize(const size_t N_ECAL,const size_t N_PS);
  treeptr _tree;
  Int_t nVtx;
  Float_t scRawEnergy, scCalibratedEnergy, scPreshowerEnergy,
    scEta, scPhi, scR, scPhiWidth, scEtaWidth, scSeedRawEnergy, 
    scSeedCalibratedEnergy, scSeedEta, scSeedPhi;
  Int_t hasParentSC, pfPhotonMatch, pfElectronMatch;
  Float_t genEnergy, genEta, genPhi, genDRToCentroid, genDRToSeed;
  Int_t N_ECALClusters;
  std::shared_ptr<Float_t> clusterRawEnergy, clusterCalibEnergy, 
    clusterEta, clusterPhi, clusterDPhiToSeed, clusterDEtaToSeed, 
    clusterDPhiToCentroid, clusterDEtaToCentroid, 
    clusterDPhiToGen, clusterDEtaToGen, clusterHitFractionSharedWithSeed;
  std::shared_ptr<Int_t> clusterInMustache, clusterInDynDPhi;
  Int_t N_PSClusters;
  std::shared_ptr<Float_t> psClusterRawEnergy, psClusterEta, psClusterPhi;
};

void PFEGCandidateTreeMaker::analyze(const edm::Event& e, 
				     const edm::EventSetup& es) {
  edm::Handle<reco::VertexCollection> vtcs;
  e.getByLabel(_vtxsrc,vtcs);
  if( vtcs.isValid() ) nVtx = vtcs->size();
  else nVtx = -1;

  edm::Handle<reco::PFCandidateCollection> pfEG;
  edm::Handle<reco::PFCandidateCollection> pfCands;
  e.getByLabel(_pfEGInput, pfEG);  
  e.getByLabel(_pfInput, pfCands);  
 
  if( pfEG.isValid() ) {
    findBestGenMatches(e, pfEG);
    for( size_t i = 0; i < pfEG->size(); ++i ) {
      processEGCandidateFillTree(e, reco::PFCandidateRef(pfEG,i), pfCands);
    }
  } else {
    throw cms::Exception("PFEGCandidateTreeMaker")
      << "Product ID for the EB SuperCluster collection was invalid!"
      << std::endl;
  }  
}

void PFEGCandidateTreeMaker::
findBestGenMatches(const edm::Event& e, 
		   const edm::Handle<reco::PFCandidateCollection>& pfs) {
  _genmatched.clear();
  reco::GenParticleRef genmatch;
  // gen information (if needed)
  if( _dogen ) {
    edm::Handle<reco::GenParticleCollection> genp;
    std::vector<reco::GenParticleRef> elesandphos;
    e.getByLabel(_geninput,genp);
    if( genp.isValid() ) {     
      reco::GenParticleRef bestmatch;
      for(size_t i = 0; i < genp->size(); ++i) {	
	const int pdgid = std::abs(genp->at(i).pdgId());
	if( pdgid == 22 || pdgid == 11 ) {	     
	  elesandphos.push_back(reco::GenParticleRef(genp,i));
	}
      }
      for( size_t i = 0; i < elesandphos.size(); ++i ) {
	double dE_min = -1;
	reco::PFCandidateRef bestmatch;
	for( size_t k = 0; k < pfs->size(); ++k ) {	  
	  reco::SuperClusterRef scref = pfs->at(k).superClusterRef();
	  if( scref.isAvailable() && scref.isNonnull() && 
	      reco::deltaR(*scref,*elesandphos[i]) < 0.3 ) {
	    double dE = std::abs(scref->energy()-elesandphos[i]->energy());
	    if( dE_min == -1 || dE < dE_min ) {
	      dE_min = dE;
	      bestmatch = reco::PFCandidateRef(pfs,k);
	    }
	  }
	}
      	_genmatched[bestmatch] = elesandphos[i];
      }
    } else {
      throw cms::Exception("PFSuperClusterTreeMaker")
      << "Requested generator level information was not available!"
      << std::endl;
    }
  }
}

void PFEGCandidateTreeMaker::
processEGCandidateFillTree(const edm::Event& e, 
		      const reco::PFCandidateRef& pf,
		      const edm::Handle<reco::PFCandidateCollection>& pfCands) {
  if( pf->superClusterRef().isNull() || !pf->superClusterRef().isAvailable() ) {
    return;
  }
  if( pf->egammaExtraRef().isNull() || !pf->egammaExtraRef().isAvailable() ) {
    return;
  }
  const reco::SuperCluster& sc = *(pf->superClusterRef());
  reco::SuperClusterRef egsc = pf->egammaExtraRef()->superClusterPFECALRef();
  bool eleMatch(getPFCandMatch(*pf,pfCands,11));
  bool phoMatch(getPFCandMatch(*pf,pfCands,22));


  const int N_ECAL = sc.clustersSize();
  const int N_PS   = sc.preshowerClustersSize();
  const double sc_eta = std::abs(sc.position().Eta());
  const double sc_cosheta = std::cosh(sc_eta);
  const double sc_pt = sc.rawEnergy()/sc_cosheta;
  if( !N_ECAL ) return;
  if( (sc_pt < 3.0 && sc_eta < 2.0) || 
      (sc_pt < 4.0 && sc_eta < 2.5 && sc_eta > 2.0) ||
      (sc_pt < 6.0 && sc_eta > 2.5) ) return;
  N_ECALClusters = std::max(0,N_ECAL - 1); // minus 1 because of seed
  N_PSClusters = N_PS;
  reco::GenParticleRef genmatch;
  // gen information (if needed)
  if( _dogen ) {    
    std::map<reco::PFCandidateRef,reco::GenParticleRef>::iterator itrmatch;
    if(  (itrmatch = _genmatched.find(pf)) != _genmatched.end() ) {
      genmatch  = itrmatch->second;           
      genEnergy = genmatch->energy();
      genEta    = genmatch->eta();
      genPhi    = genmatch->phi();
      genDRToCentroid = reco::deltaR(sc,*genmatch);
      genDRToSeed = reco::deltaR(*genmatch,**(sc.clustersBegin()));
    } else {
      genEnergy = -1.0;
      genEta    = 999.0;
      genPhi    = 999.0;
      genDRToCentroid = 999.0;
      genDRToSeed = 999.0;
    }    
  }  
  // supercluster information
  setTreeArraysForSize(N_ECALClusters,N_PSClusters);
  hasParentSC = (Int_t)(egsc.isAvailable() && egsc.isNonnull());
  pfElectronMatch = (Int_t)eleMatch;
  pfPhotonMatch = (Int_t)phoMatch;
  scRawEnergy = sc.rawEnergy();
  scCalibratedEnergy = sc.energy();
  scPreshowerEnergy = sc.preshowerEnergy();
  scEta = sc.position().Eta();
  scPhi = sc.position().Phi();
  scR   = sc.position().R();
  scPhiWidth = sc.phiWidth();
  scEtaWidth = sc.etaWidth();
  // sc seed information
  edm::Ptr<reco::PFCluster> theseed = edm::Ptr<reco::PFCluster>(sc.seed());
  GetSharedRecHitFraction fractionOfSeed(theseed);
  scSeedRawEnergy = theseed->energy();
  scSeedCalibratedEnergy = _calib->energyEm(*theseed,0.0,0.0,false);
  scSeedEta = theseed->eta();
  scSeedPhi = theseed->phi();
  // loop over all clusters that aren't the seed
  auto clusend = sc.clustersEnd();
  size_t iclus = 0;
  edm::Ptr<reco::PFCluster> pclus;
  for( auto clus = sc.clustersBegin(); clus != clusend; ++clus ) {
    pclus = edm::Ptr<reco::PFCluster>(*clus);
    if( theseed == pclus ) continue;
    clusterRawEnergy.get()[iclus] = pclus->energy();
    clusterCalibEnergy.get()[iclus] = _calib->energyEm(*pclus,0.0,0.0,false);
    clusterEta.get()[iclus] = pclus->eta();
    clusterPhi.get()[iclus] = pclus->phi();
    clusterDPhiToSeed.get()[iclus] = 
      TVector2::Phi_mpi_pi(pclus->phi() - theseed->phi());
    clusterDEtaToSeed.get()[iclus] = pclus->eta() - theseed->eta();
    clusterDPhiToCentroid.get()[iclus] = 
      TVector2::Phi_mpi_pi(pclus->phi() - sc.phi());
    clusterDEtaToCentroid.get()[iclus] = pclus->eta() - sc.eta();
    clusterDPhiToCentroid.get()[iclus] = 
      TVector2::Phi_mpi_pi(pclus->phi() - sc.phi());
    clusterDEtaToCentroid.get()[iclus] = pclus->eta() - sc.eta();
    clusterHitFractionSharedWithSeed.get()[iclus] = fractionOfSeed(pclus);
    if( _dogen && genmatch.isNonnull() ) {
      clusterDPhiToGen.get()[iclus] = 
	TVector2::Phi_mpi_pi(pclus->phi() - genmatch->phi());
      clusterDEtaToGen.get()[iclus] = pclus->eta() - genmatch->eta();
    }
    clusterInMustache.get()[iclus] = (Int_t) MK::inMustache(theseed->eta(),
							     theseed->phi(),
							     pclus->energy(),
							     pclus->eta(),
							     pclus->phi());
    clusterInDynDPhi.get()[iclus] = (Int_t)
      MK::inDynamicDPhiWindow(PFLayer::ECAL_BARREL == pclus->layer(),
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
  for( auto psclus = sc.preshowerClustersBegin(); psclus != psclusend; 
       ++psclus ) {
    ppsclus = edm::Ptr<reco::PFCluster>(*psclus);
    psClusterRawEnergy.get()[ipsclus] = ppsclus->energy();    
    psClusterEta.get()[ipsclus] = ppsclus->eta();    
    psClusterPhi.get()[ipsclus] = ppsclus->phi();
    ++ipsclus;
  }
  _tree->Fill();
}



bool PFEGCandidateTreeMaker::
getPFCandMatch(const reco::PFCandidate& cand,
	       const edm::Handle<reco::PFCandidateCollection>& pf,
	       const int pdgid_search) {
  reco::PFCandidateEGammaExtraRef egxtra = cand.egammaExtraRef();
  if( egxtra.isAvailable() && egxtra.isNonnull() ) {
    reco::SuperClusterRef scref = egxtra->superClusterPFECALRef();
    if( scref.isAvailable() && scref.isNonnull() ) {
      for( auto ipf = pf->begin(); ipf != pf->end(); ++ipf ) {
	if( std::abs(ipf->pdgId()) == pdgid_search  && pdgid_search == 11) {
	  reco::GsfTrackRef gsfref = ipf->gsfTrackRef();	    
	  reco::ElectronSeedRef sRef = gsfref->seedRef().castTo<reco::ElectronSeedRef>();
	  if( sRef.isNonnull() && sRef.isAvailable() && sRef->isEcalDriven() ) {
	    reco::SuperClusterRef temp(sRef->caloCluster().castTo<reco::SuperClusterRef>());
	    if( scref == temp ) {
	      return true;
	    }
	  }
	} else if ( std::abs(ipf->pdgId()) == 22 && pdgid_search == 22) {
	  reco::SuperClusterRef temp(ipf->superClusterRef());
	  if( scref == temp ) {
	    return true;
	  }
	}
      }
    }
  }
  return false;
}



PFEGCandidateTreeMaker::PFEGCandidateTreeMaker(const PSet& p) {
  _calib.reset(new PFEnergyCalibration());
  N_ECALClusters = 1;
  N_PSClusters   = 1;
  _tree = _fs->make<TTree>("SuperClusterTree","Dump of all available SC info");
  _tree->Branch("N_ECALClusters",&N_ECALClusters,"N_ECALClusters/I");
  _tree->Branch("N_PSClusters",&N_PSClusters,"N_PSClusters/I");
  _tree->Branch("nVtx",&nVtx,"nVtx/I");
  _tree->Branch("hasParentSC",&hasParentSC,"hasParentSC/I");
  _tree->Branch("pfPhotonMatch",&pfPhotonMatch,"pfPhotonMatch/I");
  _tree->Branch("pfElectronMatch",&pfElectronMatch,"pfElectronMatch/I");
  _tree->Branch("scRawEnergy",&scRawEnergy,"scRawEnergy/F");
  _tree->Branch("scCalibratedEnergy",&scCalibratedEnergy,
		"scCalibratedEnergy/F");
  _tree->Branch("scPreshowerEnergy",&scPreshowerEnergy,"scPreshowerEnergy/F");
  _tree->Branch("scEta",&scEta,"scEta/F");
  _tree->Branch("scPhi",&scPhi,"scPhi/F");
  _tree->Branch("scR",&scR,"scR/F");
  _tree->Branch("scPhiWidth",&scPhiWidth,"scPhiWidth/F");
  _tree->Branch("scEtaWidth",&scEtaWidth,"scEtaWidth/F");
  _tree->Branch("scSeedRawEnergy",&scSeedRawEnergy,"scSeedRawEnergy/F");
  _tree->Branch("scSeedCalibratedEnergy",&scSeedCalibratedEnergy,
		"scSeedCalibratedEnergy/F");
  _tree->Branch("scSeedEta",&scSeedEta,"scSeedEta/F");
  _tree->Branch("scSeedPhi",&scSeedPhi,"scSeedPhi/F");
  // ecal cluster information
  clusterRawEnergy.reset(new Float_t[1],array_deleter<Float_t>());
  _tree->Branch("clusterRawEnergy",clusterRawEnergy.get(),
		"clusterRawEnergy[N_ECALClusters]/F");
  clusterCalibEnergy.reset(new Float_t[1],array_deleter<Float_t>());
  _tree->Branch("clusterCalibEnergy",clusterCalibEnergy.get(),
		"clusterCalibEnergy[N_ECALClusters]/F");
  clusterEta.reset(new Float_t[1],array_deleter<Float_t>());
  _tree->Branch("clusterEta",clusterEta.get(),
		"clusterEta[N_ECALClusters]/F");
  clusterPhi.reset(new Float_t[1],array_deleter<Float_t>());
  _tree->Branch("clusterPhi",clusterPhi.get(),
		"clusterPhi[N_ECALClusters]/F");
  clusterDPhiToSeed.reset(new Float_t[1],array_deleter<Float_t>());
  _tree->Branch("clusterDPhiToSeed",clusterDPhiToSeed.get(),
		"clusterDPhiToSeed[N_ECALClusters]/F");
  clusterDEtaToSeed.reset(new Float_t[1],array_deleter<Float_t>());  
  _tree->Branch("clusterDEtaToSeed",clusterDEtaToSeed.get(),
		"clusterDEtaToSeed[N_ECALClusters]/F");  
  clusterDPhiToCentroid.reset(new Float_t[1],array_deleter<Float_t>());
  _tree->Branch("clusterDPhiToCentroid",clusterDPhiToCentroid.get(),
		"clusterDPhiToCentroid[N_ECALClusters]/F");
  clusterDEtaToCentroid.reset(new Float_t[1],array_deleter<Float_t>());
  _tree->Branch("clusterDEtaToCentroid",clusterDEtaToCentroid.get(),
		"clusterDEtaToCentroid[N_ECALClusters]/F");
  clusterHitFractionSharedWithSeed.reset(new Float_t[1],
					 array_deleter<Float_t>());
  _tree->Branch("clusterHitFractionSharedWithSeed",
		clusterHitFractionSharedWithSeed.get(),
		"clusterHitFractionSharedWithSeed[N_ECALClusters]/F");
  clusterInMustache.reset(new Int_t[1],array_deleter<Int_t>());
  _tree->Branch("clusterInMustache",clusterInMustache.get(),
		"clusterInMustache[N_ECALClusters]/I");
  clusterInDynDPhi.reset(new Int_t[1],array_deleter<Int_t>());
  _tree->Branch("clusterInDynDPhi",clusterInDynDPhi.get(),
		"clusterInDynDPhi[N_ECALClusters]/I");
  // preshower information
  psClusterRawEnergy.reset(new Float_t[1],array_deleter<Float_t>());
  _tree->Branch("psClusterRawEnergy",psClusterRawEnergy.get(),
		   "psClusterRawEnergy[N_PSClusters]/F");
  psClusterEta.reset(new Float_t[1],array_deleter<Float_t>());
  _tree->Branch("psClusterEta",psClusterEta.get(),
		"psClusterEta[N_PSClusters]/F");
  psClusterPhi.reset(new Float_t[1],array_deleter<Float_t>());
  _tree->Branch("psClusterPhi",psClusterPhi.get(),
		"psClusterPhi[N_PSClusters]/F");


  if( (_dogen = p.getUntrackedParameter<bool>("doGen",false)) ) {
    _geninput = p.getParameter<edm::InputTag>("genSrc");    
    _tree->Branch("genEta",&genEta,"genEta/F");
    _tree->Branch("genPhi",&genPhi,"genPhi/F");
    _tree->Branch("genEnergy",&genEnergy,"genEnergy/F");
    _tree->Branch("genDRToCentroid",&genDRToCentroid,"genDRToCentroid/F");
    _tree->Branch("genDRToSeed",&genDRToSeed,"genDRToSeed/F");
    
    clusterDPhiToGen.reset(new Float_t[1],array_deleter<Float_t>());
    _tree->Branch("clusterDPhiToGen",clusterDPhiToGen.get(),
		  "clusterDPhiToGen[N_ECALClusters]/F");
    clusterDEtaToGen.reset(new Float_t[1],array_deleter<Float_t>());
    _tree->Branch("clusterDEtaToGen",clusterDEtaToGen.get(),
		  "clusterDPhiToGen[N_ECALClusters]/F");
  }
  _vtxsrc    = p.getParameter<edm::InputTag>("primaryVertices");
  _pfEGInput  = p.getParameter<edm::InputTag>("pfEGammaCandSrc");
  _pfInput = p.getParameter<edm::InputTag>("pfCandSrc"); 

}


void PFEGCandidateTreeMaker::setTreeArraysForSize(const size_t N_ECAL,
						   const size_t N_PS) {
  Float_t* cRE_new = new Float_t[N_ECAL];
  clusterRawEnergy.reset(cRE_new,array_deleter<Float_t>());
  _tree->GetBranch("clusterRawEnergy")->SetAddress(clusterRawEnergy.get());
  Float_t* cCE_new = new Float_t[N_ECAL];
  clusterCalibEnergy.reset(cCE_new,array_deleter<Float_t>());
  _tree->GetBranch("clusterCalibEnergy")->SetAddress(clusterCalibEnergy.get());
  Float_t* cEta_new = new Float_t[N_ECAL];
  clusterEta.reset(cEta_new,array_deleter<Float_t>());
  _tree->GetBranch("clusterEta")->SetAddress(clusterEta.get());
  Float_t* cPhi_new = new Float_t[N_ECAL];
  clusterPhi.reset(cPhi_new,array_deleter<Float_t>());
  _tree->GetBranch("clusterPhi")->SetAddress(clusterPhi.get());
  Float_t* cDPhiSeed_new = new Float_t[N_ECAL];
  clusterDPhiToSeed.reset(cDPhiSeed_new,array_deleter<Float_t>());
  _tree->GetBranch("clusterDPhiToSeed")->SetAddress(clusterDPhiToSeed.get());
  Float_t* cDEtaSeed_new = new Float_t[N_ECAL];
  clusterDEtaToSeed.reset(cDEtaSeed_new,array_deleter<Float_t>());  
  _tree->GetBranch("clusterDEtaToSeed")->SetAddress(clusterDEtaToSeed.get());
  Float_t* cDPhiCntr_new = new Float_t[N_ECAL];
  clusterDPhiToCentroid.reset(cDPhiCntr_new,array_deleter<Float_t>());
  _tree->GetBranch("clusterDPhiToCentroid")->SetAddress(clusterDPhiToCentroid.get());
  Float_t* cDEtaCntr_new = new Float_t[N_ECAL];
  clusterDEtaToCentroid.reset(cDEtaCntr_new,array_deleter<Float_t>());
  _tree->GetBranch("clusterDEtaToCentroid")->SetAddress(clusterDEtaToCentroid.get());
  Float_t* cHitFracShared_new = new Float_t[N_ECAL];
  clusterHitFractionSharedWithSeed.reset(cHitFracShared_new,
					 array_deleter<Float_t>());
  _tree->GetBranch("clusterHitFractionSharedWithSeed")->SetAddress(clusterHitFractionSharedWithSeed.get());
  
  if( _dogen ) {
    Float_t* cDPhiGen_new = new Float_t[N_ECAL];
    clusterDPhiToGen.reset(cDPhiGen_new,array_deleter<Float_t>());
    _tree->GetBranch("clusterDPhiToGen")->SetAddress(clusterDPhiToGen.get());
    Float_t* cDEtaGen_new = new Float_t[N_ECAL];
    clusterDEtaToGen.reset(cDEtaGen_new,array_deleter<Float_t>());
    _tree->GetBranch("clusterDEtaToGen")->SetAddress(clusterDEtaToGen.get());
  }
  Int_t* cInMust_new = new Int_t[N_ECAL];
  clusterInMustache.reset(cInMust_new,array_deleter<Int_t>());
  _tree->GetBranch("clusterInMustache")->SetAddress(clusterInMustache.get());
  Int_t* cInDynDPhi_new = new Int_t[N_ECAL];
  clusterInDynDPhi.reset(cInDynDPhi_new,array_deleter<Int_t>());
  _tree->GetBranch("clusterInDynDPhi")->SetAddress(clusterInDynDPhi.get());
  Float_t* psRE_new = new Float_t[N_PS];
  psClusterRawEnergy.reset(psRE_new,array_deleter<Float_t>());
  _tree->GetBranch("psClusterRawEnergy")->SetAddress(psClusterRawEnergy.get());
  Float_t* psEta_new = new Float_t[N_PS];
  psClusterEta.reset(psEta_new,array_deleter<Float_t>());
  _tree->GetBranch("psClusterEta")->SetAddress(psClusterEta.get());
  Float_t* psPhi_new = new Float_t[N_PS];
  psClusterPhi.reset(psPhi_new,array_deleter<Float_t>());
  _tree->GetBranch("psClusterPhi")->SetAddress(psClusterPhi.get());
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFEGCandidateTreeMaker);
