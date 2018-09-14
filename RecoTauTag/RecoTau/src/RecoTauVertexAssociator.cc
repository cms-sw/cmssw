#include "RecoTauTag/RecoTau/interface/RecoTauVertexAssociator.h"

#include <functional>
#include <boost/foreach.hpp>

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include <TMath.h>

namespace reco { namespace tau {

// Get the highest pt track in a jet.
// Get the KF track if it exists.  Otherwise, see if it has a GSF track.
reco::TrackBaseRef RecoTauVertexAssociator::getLeadTrack(const PFJet& jet) const
{
  std::vector<PFCandidatePtr> chargedPFCands = pfChargedCands(jet, true);
  if ( verbosity_ >= 1 ) {
    std::cout << "<RecoTauVertexAssociator::getLeadTrack>:" << std::endl;
    std::cout << " jet: Pt = " << jet.pt() << ", eta = " << jet.eta() << ", phi = " << jet.phi() << std::endl;
    std::cout << " num. chargedPFCands = " << chargedPFCands.size() << std::endl;
    std::cout << " vxTrkFiltering = " << vxTrkFiltering_ << std::endl;
  }

  if ( chargedPFCands.empty() ) {
    return reco::TrackBaseRef();
  }

  std::vector<PFCandidatePtr> selectedPFCands;
  if ( vxTrkFiltering_ ) {
    selectedPFCands = qcuts_->filterCandRefs(chargedPFCands);
  } else { 
    selectedPFCands = chargedPFCands;
  }
  if ( verbosity_ >= 1 ) {
    std::cout << " num. selectedPFCands = " << selectedPFCands.size() << std::endl;
  }

  PFCandidatePtr leadPFCand;
  if ( !selectedPFCands.empty() ) {
    double leadTrackPt = 0.;
    if ( leadingTrkOrPFCandOption_ == kFirstTrack){ leadPFCand=selectedPFCands[0];}
    else
    {
      for ( std::vector<PFCandidatePtr>::const_iterator pfCand = selectedPFCands.begin();
  	  pfCand != selectedPFCands.end(); ++pfCand ) {
        const reco::Track* track = nullptr;
        if ( (*pfCand)->trackRef().isNonnull() ) track = (*pfCand)->trackRef().get();
        else if ( (*pfCand)->gsfTrackRef().isNonnull() ) track = (*pfCand)->gsfTrackRef().get();
        if ( !track ) continue;
        double trackPt = 0.;
        if ( leadingTrkOrPFCandOption_ == kLeadTrack ) {
  	  //double trackPt = track->pt();
  	  trackPt = track->pt() - 2.*track->ptError();
        } else if ( leadingTrkOrPFCandOption_ == kLeadPFCand ) {
	  trackPt = (*pfCand)->pt();
        } else if ( leadingTrkOrPFCandOption_ == kMinLeadTrackOrPFCand ) {
	  trackPt = TMath::Min(track->pt(), (double)(*pfCand)->pt());
	} else assert(0);
        if ( trackPt > leadTrackPt ) {
          leadPFCand = (*pfCand);
	  leadTrackPt = trackPt;
        }
      }
    }
  }
  if ( leadPFCand.isNull() ) {
    if ( recoverLeadingTrk_ ) {
      leadPFCand = chargedPFCands[0];
    } else {
      return reco::TrackBaseRef();
    } 
  }
  if ( verbosity_ >= 1 ) {
    std::cout << "leadPFCand: Pt = " << leadPFCand->pt() << ", eta = " << leadPFCand->eta() << ", phi = " << leadPFCand->phi() << std::endl;
  }
  
  if ( leadPFCand->trackRef().isNonnull() ) {
    return reco::TrackBaseRef(leadPFCand->trackRef());
  } else if ( leadPFCand->gsfTrackRef().isNonnull() ) {
    return reco::TrackBaseRef(leadPFCand->gsfTrackRef());
  } 
  return reco::TrackBaseRef();
}

namespace {
  // Define functors which extract the relevant information from a collection of
  // vertices.
  double dzToTrack(const reco::VertexRef& vtx, const reco::TrackBaseRef& trk)
  {
    if ( !trk || !vtx ) {
      return std::numeric_limits<double>::infinity();
    }
    return std::abs(trk->dz(vtx->position()));
  }
  
  double trackWeightInVertex(const reco::VertexRef& vtx, const reco::TrackBaseRef& trk)
  {
    if ( !trk || !vtx ) {
      return 0.0;
    }
    return vtx->trackWeight(trk);
  }
}

RecoTauVertexAssociator::RecoTauVertexAssociator(const edm::ParameterSet& pset, edm::ConsumesCollector && iC)
  : vertexSelector_(nullptr),
    qcuts_(nullptr),
    jetToVertexAssociation_(nullptr),
    lastEvent_(0)
{
  //std::cout << "<RecoTauVertexAssociator::RecoTauVertexAssociator>:" << std::endl;

  vertexTag_ = edm::InputTag("offlinePrimaryVertices", "");
  algorithm_ = "highestPtInEvent";
  // Sanity check, will remove once HLT module configs are updated.
  if ( !pset.exists("primaryVertexSrc") || !pset.exists("pvFindingAlgo") ) {
    edm::LogWarning("NoVertexFindingMethodSpecified")
      << "The PSet passed to the RecoTauVertexAssociator was incorrectly configured."
      << " The vertex will be taken as the highest Pt vertex from the 'offlinePrimaryVertices' collection."
      << std::endl;
  } else {
    vertexTag_ = pset.getParameter<edm::InputTag>("primaryVertexSrc");
    algorithm_ = pset.getParameter<std::string>("pvFindingAlgo");
  }
    
  if ( pset.exists("vxAssocQualityCuts") ) {
    //std::cout << " reading 'vxAssocQualityCuts'" << std::endl;
    qcuts_ = new RecoTauQualityCuts(pset.getParameterSet("vxAssocQualityCuts"));
  } else {
    //std::cout << " reading 'signalQualityCuts'" << std::endl;
    qcuts_ = new RecoTauQualityCuts(pset.getParameterSet("signalQualityCuts"));
  }
  assert(qcuts_);

  vxTrkFiltering_ = false;
  if ( !pset.exists("vertexTrackFiltering") && pset.exists("vxAssocQualityCuts") ) {
    edm::LogWarning("NoVertexTrackFilteringSpecified")
      << "The PSet passed to the RecoTauVertexAssociator was incorrectly configured." 
      << " Please define vertexTrackFiltering in config file." 
      << " No filtering of tracks to vertices will be applied."
      << std::endl;
  } else {
    vxTrkFiltering_ = pset.exists("vertexTrackFiltering") ? 
      pset.getParameter<bool>("vertexTrackFiltering") : false;
  }
  if ( pset.exists("vertexSelection") ) {
    std::string vertexSelection = pset.getParameter<std::string>("vertexSelection");
    if ( !vertexSelection.empty() ) {
      vertexSelector_ = new StringCutObjectSelector<reco::Vertex>(vertexSelection);
    }
  }
  
  if ( algorithm_ == "highestPtInEvent" ) {
    algo_ = kHighestPtInEvent;
  } else if ( algorithm_ == "closestInDeltaZ" ) {
    algo_ = kClosestDeltaZ;
  } else if ( algorithm_ == "highestWeightForLeadTrack" ) {
    algo_ = kHighestWeigtForLeadTrack;
  } else if ( algorithm_ == "combined" ) {
    algo_ = kCombined;
  } else {
    throw cms::Exception("BadVertexAssociatorConfig")
      << "Invalid Configuration parameter 'algorithm' " << algorithm_ << "." 
      << " Valid options are: 'highestPtInEvent', 'closestInDeltaZ', 'highestWeightForLeadTrack' and 'combined'.\n";
  }

  vxToken_ = iC.consumes<reco::VertexCollection>(vertexTag_);

  recoverLeadingTrk_ = pset.exists("recoverLeadingTrk") ? 
    pset.getParameter<bool>("recoverLeadingTrk") : false;

  std::string leadingTrkOrPFCandOption_string = pset.exists("leadingTrkOrPFCandOption") ? pset.getParameter<std::string>("leadingTrkOrPFCandOption") : "firstTrack" ;
  if      ( leadingTrkOrPFCandOption_string == "leadTrack"  ) leadingTrkOrPFCandOption_ = kLeadTrack;
  else if ( leadingTrkOrPFCandOption_string == "leadPFCand" ) leadingTrkOrPFCandOption_ = kLeadPFCand;
  else if ( leadingTrkOrPFCandOption_string == "minLeadTrackOrPFCand" ) leadingTrkOrPFCandOption_ = kMinLeadTrackOrPFCand;
  else if ( leadingTrkOrPFCandOption_string == "firstTrack" ) leadingTrkOrPFCandOption_ = kFirstTrack;
  else throw cms::Exception("BadVertexAssociatorConfig")
    << "Invalid Configuration parameter 'leadingTrkOrPFCandOption' " << leadingTrkOrPFCandOption_string << "." 
    << " Valid options are: 'leadTrack', 'leadPFCand', 'firstTrack'.\n";
  
  verbosity_ = ( pset.exists("verbosity") ) ?
    pset.getParameter<int>("verbosity") : 0;
}

RecoTauVertexAssociator::~RecoTauVertexAssociator()
{ 
  delete vertexSelector_; 
  delete qcuts_;
  delete jetToVertexAssociation_;
}

void RecoTauVertexAssociator::setEvent(const edm::Event& evt) 
{
  edm::Handle<reco::VertexCollection> vertices;
  evt.getByToken(vxToken_, vertices);
  selectedVertices_.clear();
  selectedVertices_.reserve(vertices->size());
  for ( size_t idxVertex = 0; idxVertex < vertices->size(); ++idxVertex ) {
    reco::VertexRef vertex(vertices, idxVertex);
    if ( vertexSelector_ && !(*vertexSelector_)(*vertex) ) continue;
    selectedVertices_.push_back(vertex);
  }
  if ( !selectedVertices_.empty() ) {
    qcuts_->setPV(selectedVertices_[0]);
  }
  edm::EventNumber_t currentEvent = evt.id().event();
  if ( currentEvent != lastEvent_ || !jetToVertexAssociation_ ) {
    if ( !jetToVertexAssociation_ ) jetToVertexAssociation_ = new std::map<const reco::PFJet*, reco::VertexRef>;
    else jetToVertexAssociation_->clear();
    lastEvent_ = currentEvent;
  }
}

reco::VertexRef
RecoTauVertexAssociator::associatedVertex(const PFTau& tau, bool useJet) const 
{

  if ( !useJet ) {
    if ( tau.leadPFChargedHadrCand().isNonnull() ) {
      if ( tau.leadPFChargedHadrCand()->trackRef().isNonnull() )
	return associatedVertex( reco::TrackBaseRef( tau.leadPFChargedHadrCand()->trackRef() ) );
      else if (  tau.leadPFChargedHadrCand()->gsfTrackRef().isNonnull() )
	return associatedVertex( reco::TrackBaseRef( tau.leadPFChargedHadrCand()->gsfTrackRef() ) );
    }
  }
  // MB: use vertex associated to a given jet if explicitely requested or in case of missing leading track
  reco::PFJetRef jetRef = tau.jetRef();
  // FIXME workaround for HLT which does not use updated data format
  if ( jetRef.isNull() ) jetRef = tau.pfTauTagInfoRef()->pfjetRef();
  return associatedVertex(*jetRef);
}

reco::VertexRef
RecoTauVertexAssociator::associatedVertex(const TrackBaseRef& track) const 
{

  reco::VertexRef trkVertex = ( !selectedVertices_.empty() ) ? selectedVertices_[0] : reco::VertexRef();

  if ( algo_ == kHighestPtInEvent ) {
    if ( !selectedVertices_.empty() ) trkVertex = selectedVertices_[0];
  } else if ( algo_ == kClosestDeltaZ ) {
    if ( track.isNonnull() ) {
      double closestDistance = 1.e+6;
      // Find the vertex that has the lowest dZ to the track
      int idxVertex = 0;
      for ( std::vector<reco::VertexRef>::const_iterator selectedVertex = selectedVertices_.begin();
	    selectedVertex != selectedVertices_.end(); ++selectedVertex ) {
	double dZ = dzToTrack(*selectedVertex, track);
	if ( verbosity_ ) {
	  std::cout << "vertex #" << idxVertex << ": x = " << (*selectedVertex)->position().x() << ", y = " << (*selectedVertex)->position().y() << ", z = " << (*selectedVertex)->position().z() 
		    << " --> dZ = " << dZ << std::endl;
	}
	if ( dZ < closestDistance ) {
	  trkVertex = (*selectedVertex);
	  closestDistance = dZ;
	}
	++idxVertex;
      }
    }
  } else if ( algo_ == kHighestWeigtForLeadTrack || algo_ == kCombined ) {
    if ( track.isNonnull() ) {
      double largestWeight = -1.;
      // Find the vertex that has the highest association probability to the track
      int idxVertex = 0;
      for ( std::vector<reco::VertexRef>::const_iterator selectedVertex = selectedVertices_.begin();
	    selectedVertex != selectedVertices_.end(); ++selectedVertex ) {
	double weight = trackWeightInVertex(*selectedVertex, track);
	if ( verbosity_ ) {
	  std::cout << "vertex #" << idxVertex << ": x = " << (*selectedVertex)->position().x() << ", y = " << (*selectedVertex)->position().y() << ", z = " << (*selectedVertex)->position().z() 
		    << " --> weight = " << weight << std::endl;
	}
	if ( weight > largestWeight ) {
	  trkVertex = (*selectedVertex);
	  largestWeight = weight;
	}
	++idxVertex;
      }
      // the weight was never larger than zero
      if ( algo_ == kCombined && largestWeight < 1.e-7 ) {
	if ( verbosity_ ) {
	  std::cout << "No vertex had positive weight! Trying dZ instead... " << std::endl;
	}
	double closestDistance = 1.e+6;
	// Find the vertex that has the lowest dZ to the leading track
	int idxVertex = 0;
	for ( std::vector<reco::VertexRef>::const_iterator selectedVertex = selectedVertices_.begin();
	      selectedVertex != selectedVertices_.end(); ++selectedVertex ) {
	  double dZ = dzToTrack(*selectedVertex, track);
	  if ( verbosity_ ) {
	    std::cout << "vertex #" << idxVertex << ": x = " << (*selectedVertex)->position().x() << ", y = " << (*selectedVertex)->position().y() << ", z = " << (*selectedVertex)->position().z() 
		      << " --> dZ = " << dZ << std::endl;
	  }
	  if ( dZ < closestDistance ) {
	    trkVertex = (*selectedVertex);
	    closestDistance = dZ;
	  }
	  ++idxVertex;
	}
      }
    }
  }

  if ( verbosity_ >= 1 ) {
    std::cout << "--> returning vertex: x = " << trkVertex->position().x() << ", y = " << trkVertex->position().y() << ", z = " << trkVertex->position().z() << std::endl;
  }
  
  return trkVertex;
}

reco::VertexRef
RecoTauVertexAssociator::associatedVertex(const PFJet& jet) const 
{
  if ( verbosity_ >= 1 ) {
    std::cout << "<RecoTauVertexAssociator::associatedVertex>:" << std::endl;
    std::cout << " jet: Pt = " << jet.pt() << ", eta = " << jet.eta() << ", phi = " << jet.phi() << std::endl;
    std::cout << " num. Vertices = " << selectedVertices_.size() << std::endl;
    std::cout << " size(jetToVertexAssociation) = " << jetToVertexAssociation_->size() << std::endl;
    std::cout << " vertexTag = " << vertexTag_ << std::endl;
    std::cout << " algorithm = " << algorithm_ << std::endl;
    std::cout << " recoverLeadingTrk = " << recoverLeadingTrk_ << std::endl;
  }

  reco::VertexRef jetVertex = ( !selectedVertices_.empty() ) ? selectedVertices_[0] : reco::VertexRef();
  const PFJet* jetPtr = &jet;

  // check if jet-vertex association has been determined for this jet before
  std::map<const reco::PFJet*, reco::VertexRef>::iterator vertexPtr = jetToVertexAssociation_->find(jetPtr);
  if ( vertexPtr != jetToVertexAssociation_->end() ) {
    jetVertex = vertexPtr->second;
  } else {
    // no jet-vertex association exists for this jet yet, compute it!
    if ( algo_ == kHighestPtInEvent ) {
      if ( !selectedVertices_.empty() ) jetVertex = selectedVertices_[0];
    } else if ( algo_ == kClosestDeltaZ || 
		algo_ == kHighestWeigtForLeadTrack || 
		algo_ == kCombined ) {
      // find "leading" (highest Pt) track in jet
      reco::TrackBaseRef leadTrack = getLeadTrack(jet);
      if ( verbosity_ ) {
	if ( leadTrack.isNonnull() ) std::cout << "leadTrack: Pt = " << leadTrack->pt() << ", eta = " << leadTrack->eta() << ", phi = " << leadTrack->phi() << std::endl;
	else std::cout << "leadTrack: N/A" << std::endl;
      }
      if ( leadTrack.isNonnull() ) {
	jetVertex = associatedVertex(leadTrack);
      }
    }
    
    jetToVertexAssociation_->insert(std::pair<const PFJet*, reco::VertexRef>(jetPtr, jetVertex));
  }

  if ( verbosity_ >= 1 ) {
    std::cout << "--> returning vertex: x = " << jetVertex->position().x() << ", y = " << jetVertex->position().y() << ", z = " << jetVertex->position().z() << std::endl;
  }
  
  return jetVertex;
}

}}
