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

#include "DataFormats/ParticleFlowReco/interface/PFBlockElementGsfTrack.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementSuperCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElementBrem.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"

#include <algorithm>
#include <memory>

namespace {
  struct ElementEquals {
    const reco::PFBlockElement& cmpElem;
    ElementEquals(const reco::PFBlockElement& e) : cmpElem(e) {}
    bool operator()(const reco::PFBlockElement& chkElem ) const {
      bool result = false;
      if( cmpElem.type() == chkElem.type() ) {
	switch( cmpElem.type() ) {
	case reco::PFBlockElement::TRACK:
	  result = ( cmpElem.trackRef().isNonnull() &&
		     chkElem.trackRef().isNonnull() &&
		     cmpElem.trackRef()->momentum() == 
		     chkElem.trackRef()->momentum() );	    
	  break;
	case reco::PFBlockElement::PS1:
	case reco::PFBlockElement::PS2:
	case reco::PFBlockElement::ECAL:
	case reco::PFBlockElement::HCAL:
	case reco::PFBlockElement::HFEM:
	case reco::PFBlockElement::HFHAD:
	case reco::PFBlockElement::HO:
	  result = ( cmpElem.clusterRef().isNonnull() &&
		     chkElem.clusterRef().isNonnull() &&
		     cmpElem.clusterRef().key() == chkElem.clusterRef().key() );	    
	  break;
	case reco::PFBlockElement::SC:
	  {
	    const reco::PFBlockElementSuperCluster& cmpSC= 
	      static_cast<const reco::PFBlockElementSuperCluster&>(cmpElem);
	    const reco::PFBlockElementSuperCluster& chkSC= 
	      static_cast<const reco::PFBlockElementSuperCluster&>(chkElem);	      
	    result = ( cmpSC.superClusterRef().isNonnull() &&
		       chkSC.superClusterRef().isNonnull() &&
		       cmpSC.superClusterRef()->position() == 
		       chkSC.superClusterRef()->position() );	      
	  }
	  break;
	case reco::PFBlockElement::GSF:
	  {
	    const reco::PFBlockElementGsfTrack& cmpGSF = 
	      static_cast<const reco::PFBlockElementGsfTrack&>(cmpElem);
	    const reco::PFBlockElementGsfTrack& chkGSF = 
	      static_cast<const reco::PFBlockElementGsfTrack&>(chkElem);	      
	    result = ( cmpGSF.GsftrackRef().isNonnull() &&
		       chkGSF.GsftrackRef().isNonnull() &&
		       cmpGSF.GsftrackRef()->momentum() == 
		       chkGSF.GsftrackRef()->momentum() );	      
	  }
	  break;
	case reco::PFBlockElement::BREM:
	  {
	    const reco::PFBlockElementBrem& cmpBREM = 
	      static_cast<const reco::PFBlockElementBrem&>(cmpElem);
	    const reco::PFBlockElementBrem& chkBREM = 
	      static_cast<const reco::PFBlockElementBrem&>(chkElem);	      
	    result = ( cmpBREM.GsftrackRef().isNonnull() &&
		       chkBREM.GsftrackRef().isNonnull() &&
		       cmpBREM.GsftrackRef()->momentum() == 
		       chkBREM.GsftrackRef()->momentum() &&
		       cmpBREM.indTrajPoint() == chkBREM.indTrajPoint() );	      
	  }
	  break;
	default:
	  throw cms::Exception("UnknownBlockElementType")
	    << cmpElem.type() << " is not a known PFBlockElement type!" << std::endl;
	}
      }
      return result;
    }
  };

  struct HasElementOverlap {
    const reco::PFBlock* tocompare;
    HasElementOverlap(const reco::PFBlock& a) : tocompare(&a) {}
    bool operator()(const reco::PFBlock& b) const {
      bool result = false;     
      for( const auto& cmpElem : tocompare->elements() ) {		  
	for( const auto& chkElem : b.elements() ) {	  
	  // can't be equal elements if type not equal
	  if( cmpElem.type() != chkElem.type() ) continue; 
	  switch( cmpElem.type() ) {
	  case reco::PFBlockElement::TRACK:
	    result = ( cmpElem.trackRef().isNonnull() &&
		       chkElem.trackRef().isNonnull() &&
		       cmpElem.trackRef()->momentum() == chkElem.trackRef()->momentum() );	    
	    break;
	  case reco::PFBlockElement::PS1:
	  case reco::PFBlockElement::PS2:
	  case reco::PFBlockElement::ECAL:
	  case reco::PFBlockElement::HCAL:
	  case reco::PFBlockElement::HFEM:
	  case reco::PFBlockElement::HFHAD:
	  case reco::PFBlockElement::HO:
	    result = ( cmpElem.clusterRef().isNonnull() &&
		       chkElem.clusterRef().isNonnull() &&
		       cmpElem.clusterRef().key() == chkElem.clusterRef().key() );	    
	    break;
	  case reco::PFBlockElement::SC:
	    {
	      const reco::PFBlockElementSuperCluster& cmpSC= 
		static_cast<const reco::PFBlockElementSuperCluster&>(cmpElem);
	      const reco::PFBlockElementSuperCluster& chkSC= 
		static_cast<const reco::PFBlockElementSuperCluster&>(chkElem);	      
	      result = ( cmpSC.superClusterRef().isNonnull() &&
			 chkSC.superClusterRef().isNonnull() &&
			 cmpSC.superClusterRef()->position() == 
			 chkSC.superClusterRef()->position() );	      
	    }
	    break;
	  case reco::PFBlockElement::GSF:
	    {
	      const reco::PFBlockElementGsfTrack& cmpGSF = 
		static_cast<const reco::PFBlockElementGsfTrack&>(cmpElem);
	      const reco::PFBlockElementGsfTrack& chkGSF = 
		static_cast<const reco::PFBlockElementGsfTrack&>(chkElem);	      
	      result = ( cmpGSF.GsftrackRef().isNonnull() &&
			 chkGSF.GsftrackRef().isNonnull() &&
			 cmpGSF.GsftrackRef()->momentum() == chkGSF.GsftrackRef()->momentum() );	      
	    }
	    break;
	  case reco::PFBlockElement::BREM:
	    {
	      const reco::PFBlockElementBrem& cmpBREM = 
		static_cast<const reco::PFBlockElementBrem&>(cmpElem);
	      const reco::PFBlockElementBrem& chkBREM = 
		static_cast<const reco::PFBlockElementBrem&>(chkElem);	      
	      result = ( cmpBREM.GsftrackRef().isNonnull() &&
			 chkBREM.GsftrackRef().isNonnull() &&
			 cmpBREM.GsftrackRef()->momentum() == chkBREM.GsftrackRef()->momentum() &&
			 cmpBREM.indTrajPoint() == chkBREM.indTrajPoint() );	      
	    }
	    break;
	  default:
	    throw cms::Exception("UnknownBlockElementType")
	      << cmpElem.type() << " is not a known PFBlockElement type!" << std::endl;
	  }
	  if( result ) break;
	}
	if( result ) break;
      }
      return result;
    }
  };
}

typedef edm::ParameterSet PSet;

class PFBlockComparator : public edm::EDAnalyzer {
public:
  PFBlockComparator(const PSet& c) : 
    _src(c.getParameter<edm::InputTag>("source")),
    _srcOld(c.getParameter<edm::InputTag>("sourceOld")){};
  ~PFBlockComparator() {}

  void analyze(const edm::Event&, const edm::EventSetup&);
private:    
  edm::InputTag _src;
  edm::InputTag _srcOld;
};

void PFBlockComparator::analyze(const edm::Event& e, 
				const edm::EventSetup& es) {
  edm::Handle<reco::PFBlockCollection> blocks;
  e.getByLabel(_src,blocks);
  edm::Handle<reco::PFBlockCollection> oldblocks;
  e.getByLabel(_srcOld,oldblocks);
  
  unsigned matchedblocks = 0;

  if( blocks->size() != oldblocks->size() ) {
    std::cout << "+++WARNING+++ Number of new blocks does not match number of old blocks!!!!" << std::endl;
    std::cout << "              " << blocks->size() << " != " << oldblocks->size() << std::endl;
  }

  for( const auto& block : *blocks ) {
    HasElementOverlap checker(block);
    auto matched_block = std::find_if(oldblocks->begin(),oldblocks->end(),checker);
    if( matched_block != oldblocks->end() ) {
      ++matchedblocks;
      if( block.elements().size() != matched_block->elements().size() ) {
	std::cout << "Number of elements in the block is not the same!" << std::endl;
	std::cout << block.elements().size() << ' ' << matched_block->elements().size() << std::endl;
      }
      if( block.linkData().size() != matched_block->linkData().size() ) {
	std::cout << "Something is really fucked up, captain..." << std::endl;
	std::cout << block.elements().size() << ' ' << matched_block->elements().size() << std::endl;
      }
      unsigned found_elements = 0;
      for( const auto& elem : block.elements() ) {
	ElementEquals ele_check(elem);	
	auto matched_elem = std::find_if(matched_block->elements().begin(),
					 matched_block->elements().end(),ele_check);
	if( matched_elem != matched_block->elements().end() ) {
	  ++found_elements;
	  std::multimap<double,unsigned> new_elems, old_elems;
	  block.associatedElements(elem.index(),block.linkData(),new_elems);
	  matched_block->associatedElements(matched_elem->index(),matched_block->linkData(),old_elems);
	  
	  if(new_elems.size() == old_elems.size()) {
	    for(auto newitr = new_elems.begin(), olditr = old_elems.begin(); 
		newitr != new_elems.end(); ++newitr, ++olditr ) {	      
	      if( newitr->first != olditr->first ||
		  block.elements()[newitr->second].type() != matched_block->elements()[olditr->second].type() ) {
		std::cout << "( " << newitr->first << " , " << block.elements()[newitr->second].type() << " ) != ( "
			  << olditr->first << " , " << matched_block->elements()[olditr->second].type() << " )" << std::endl;
	      }
	    }
	  }
	  if(new_elems.size() != old_elems.size()) {
	    std::cout << "block links for element " << elem.index() << " are different from old links!" << std::endl;
	    std::cout << "new: ";
	    for(auto newassc : new_elems ) { 
	      std::cout << "( " << newassc.first << " , " 
			<< newassc.second << " , " 
			<< block.elements()[newassc.second].type() << " ), "; 
	    }
	    std::cout << std::endl;
	    std::cout << "old: ";
	    for(auto oldassc : old_elems ) { 
	      std::cout << "( " << oldassc.first << " , " 
			<< oldassc.second << " , "
			<< matched_block->elements()[oldassc.second].type() << " ), "; 
	    }
	    std::cout << std::endl;	      
	  }
	} else {
	  std::cout << "+++WARNING+++ : couldn't find match for element: " << elem << std::endl;
	}
      }     
      if( found_elements != block.elements().size() ||
	  found_elements != matched_block->elements().size() ) {
	std::cout << "+++WARNING+++ : couldn't find all elements in block with " 
		  << block.elements().size() << " elements matched to block with " 
		  << matched_block->elements().size() << "!" << std::endl;
	std::cout << "new block: " << std::endl;
	std::cout << block << std::endl;
	std::cout << "old block: " << std::endl;
	std::cout << *matched_block << std::endl;
      }
    } else {
      std::cout << "+++WARNING+++ : couldn't find another block that matched the new block!" << std::endl;
      std::cout << block << std::endl;
    }
  } // block
  if( matchedblocks != oldblocks->size() ) {
    std::cout << "Wasn't able to match all blocks!" << std::endl;
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFBlockComparator);
