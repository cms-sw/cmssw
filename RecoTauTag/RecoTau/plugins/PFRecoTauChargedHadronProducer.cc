/*
 * PFRecoTauChargedHadronProducer
 *
 * Author: Christian Veelken, LLR
 *
 * Associates reconstructed ChargedHadrons to PFJets.  The ChargedHadrons are built using one
 * or more RecoTauBuilder plugins.  Any overlaps (ChargedHadrons sharing tracks)
 * are removed, with the best ChargedHadron candidates taken.  The 'best' are defined
 * via the input list of PFRecoTauChargedHadronQualityPlugins, which form a
 * lexicograpical ranking.
 *
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTauTag/RecoTau/interface/PFRecoTauChargedHadronPlugins.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCleaningTools.h"
#include "RecoTauTag/RecoTau/interface/RecoTauCommonUtilities.h"
#include "RecoTauTag/RecoTau/interface/pfRecoTauChargedHadronAuxFunctions.h"

#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/TauReco/interface/PFJetChargedHadronAssociation.h"
#include "DataFormats/TauReco/interface/PFRecoTauChargedHadron.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/ptr_container/ptr_list.hpp>

#include <string>
#include <vector>
#include <list>
#include <set>
#include <algorithm>
#include <functional>
#include <cmath>

class PFRecoTauChargedHadronProducer : public edm::stream::EDProducer<> 
{
public:
  typedef reco::tau::PFRecoTauChargedHadronBuilderPlugin Builder;
  typedef reco::tau::PFRecoTauChargedHadronQualityPlugin Ranker;

  explicit PFRecoTauChargedHadronProducer(const edm::ParameterSet& cfg);
  ~PFRecoTauChargedHadronProducer() override {}
  void produce(edm::Event& evt, const edm::EventSetup& es) override;
  template <typename T>
  void print(const T& chargedHadrons);

 private:
  typedef boost::ptr_vector<Builder> builderList;
  typedef boost::ptr_vector<Ranker> rankerList;
  typedef boost::ptr_vector<reco::PFRecoTauChargedHadron> ChargedHadronVector;
  typedef boost::ptr_list<reco::PFRecoTauChargedHadron> ChargedHadronList;

  typedef reco::tau::RecoTauLexicographicalRanking<rankerList, reco::PFRecoTauChargedHadron> ChargedHadronPredicate;

  std::string moduleLabel_;

  // input jet collection
  edm::InputTag srcJets_;
  edm::EDGetTokenT<reco::CandidateView> Jets_token;
  double minJetPt_;
  double maxJetAbsEta_;

  // plugins for building and ranking ChargedHadron candidates
  builderList builders_;
  rankerList rankers_;

  std::auto_ptr<ChargedHadronPredicate> predicate_;

  // output selector
  std::auto_ptr<StringCutObjectSelector<reco::PFRecoTauChargedHadron> > outputSelector_;

  // flag to enable/disable debug print-out
  int verbosity_;
};

PFRecoTauChargedHadronProducer::PFRecoTauChargedHadronProducer(const edm::ParameterSet& cfg) 
  : moduleLabel_(cfg.getParameter<std::string>("@module_label"))
{
  srcJets_ = cfg.getParameter<edm::InputTag>("jetSrc");
  Jets_token = consumes<reco::CandidateView>(srcJets_);
  minJetPt_ = ( cfg.exists("minJetPt") ) ? cfg.getParameter<double>("minJetPt") : -1.0;
  maxJetAbsEta_ = ( cfg.exists("maxJetAbsEta") ) ? cfg.getParameter<double>("maxJetAbsEta") : 99.0;
  verbosity_ = ( cfg.exists("verbosity") ) ?
    cfg.getParameter<int>("verbosity") : 0;
  
  // get set of ChargedHadron builder plugins
  edm::VParameterSet psets_builders = cfg.getParameter<edm::VParameterSet>("builders");
  for ( edm::VParameterSet::const_iterator pset = psets_builders.begin();
	pset != psets_builders.end(); ++pset ) {
    std::string pluginType = pset->getParameter<std::string>("plugin");
    edm::ParameterSet pset_modified = (*pset);
    pset_modified.addParameter<int>("verbosity", verbosity_);
    builders_.push_back(PFRecoTauChargedHadronBuilderPluginFactory::get()->create(pluginType, pset_modified, consumesCollector()));
  }
  
  // get set of plugins for ranking ChargedHadrons in quality
  edm::VParameterSet psets_rankers = cfg.getParameter<edm::VParameterSet>("ranking");
  for ( edm::VParameterSet::const_iterator pset = psets_rankers.begin();
	pset != psets_rankers.end(); ++pset ) {
    std::string pluginType = pset->getParameter<std::string>("plugin");
    edm::ParameterSet pset_modified = (*pset);
    pset_modified.addParameter<int>("verbosity", verbosity_);
    rankers_.push_back(PFRecoTauChargedHadronQualityPluginFactory::get()->create(pluginType, pset_modified));
  }

  // build the sorting predicate
  predicate_ = std::auto_ptr<ChargedHadronPredicate>(new ChargedHadronPredicate(rankers_));
  
  // check if we want to apply a final output selection
  if ( cfg.exists("outputSelection") ) {
    std::string selection = cfg.getParameter<std::string>("outputSelection");
    if ( !selection.empty() ) {
      outputSelector_.reset(new StringCutObjectSelector<reco::PFRecoTauChargedHadron>(selection));
    }
  }

  produces<reco::PFJetChargedHadronAssociation>();
}

void PFRecoTauChargedHadronProducer::produce(edm::Event& evt, const edm::EventSetup& es) 
{
  if ( verbosity_ ) {
    edm::LogPrint("PFRecoTauChHProducer")<< "<PFRecoTauChargedHadronProducer::produce>:" ;
    edm::LogPrint("PFRecoTauChHProducer")<< " moduleLabel = " << moduleLabel_ ;
  }

  // give each of our plugins a chance at doing something with the edm::Event
  for(auto& builder : builders_ ) {
    builder.setup(evt, es);
  }
  
  // get a view of our jets via the base candidates
  edm::Handle<reco::CandidateView> jets;
  evt.getByToken(Jets_token, jets);
  
  // convert the view to a RefVector of actual PFJets
  reco::PFJetRefVector pfJets = reco::tau::castView<reco::PFJetRefVector>(jets);

  // make our association
  std::unique_ptr<reco::PFJetChargedHadronAssociation> pfJetChargedHadronAssociations;


  if ( !pfJets.empty() ) {
    edm::Handle<reco::PFJetCollection> pfJetCollectionHandle;
    evt.get(pfJets.id(), pfJetCollectionHandle);
    pfJetChargedHadronAssociations = std::make_unique<reco::PFJetChargedHadronAssociation>(reco::PFJetRefProd(pfJetCollectionHandle));
  } else {
    pfJetChargedHadronAssociations = std::make_unique<reco::PFJetChargedHadronAssociation>();
  }

  // loop over our jets
  for(auto const& pfJet : pfJets ) {
    
    if(pfJet->pt() - minJetPt_ < 1e-5) continue;
    if(std::abs(pfJet->eta()) - maxJetAbsEta_ > -1e-5) continue;

    // build global list of ChargedHadron candidates for each jet
    ChargedHadronList uncleanedChargedHadrons;

    // merge candidates reconstructed by all desired algorithm plugins
    for(auto const& builder : builders_ ) {
      try {
        ChargedHadronVector result(builder(*pfJet));
	if ( verbosity_ ) {
	  edm::LogPrint("PFRecoTauChHProducer")<< "result of builder = " << builder.name() << ":" ;
	  print(result);
	}
        uncleanedChargedHadrons.transfer(uncleanedChargedHadrons.end(), result);
      } catch ( cms::Exception& exception ) {
        edm::LogError("BuilderPluginException")
            << "Exception caught in builder plugin " << builder.name()
            << ", rethrowing" << std::endl;
        throw exception;
      }
    }

    // rank the candidates according to our quality plugins
    uncleanedChargedHadrons.sort(*predicate_);

    // define collection of cleaned ChargedHadrons;
    std::vector<reco::PFRecoTauChargedHadron> cleanedChargedHadrons;

    // keep track of neutral PFCandidates, charged PFCandidates and tracks "used" by ChargedHadron candidates in the clean collection
    typedef std::pair<double, double> etaPhiPair;
    std::list<etaPhiPair> tracksInCleanCollection;
    std::set<reco::PFCandidatePtr> neutralPFCandsInCleanCollection;

    while ( !uncleanedChargedHadrons.empty() ) {
      
      // get next best ChargedHadron candidate
      std::auto_ptr<reco::PFRecoTauChargedHadron> nextChargedHadron(uncleanedChargedHadrons.pop_front().release());
      if ( verbosity_ ) {
	edm::LogPrint("PFRecoTauChHProducer")<< "processing nextChargedHadron:" ;
	edm::LogPrint("PFRecoTauChHProducer")<< (*nextChargedHadron);
      }

      // discard candidates which fail final output selection
      if ( !(*outputSelector_)(*nextChargedHadron) ) continue;

      const reco::Track* track = nullptr;
      if ( nextChargedHadron->getChargedPFCandidate().isNonnull() ) {
	const reco::PFCandidatePtr& chargedPFCand = nextChargedHadron->getChargedPFCandidate();
	if ( chargedPFCand->trackRef().isNonnull() ) track = chargedPFCand->trackRef().get();
	else if ( chargedPFCand->muonRef().isNonnull() && chargedPFCand->muonRef()->innerTrack().isNonnull()  ) track = chargedPFCand->muonRef()->innerTrack().get();
	else if ( chargedPFCand->muonRef().isNonnull() && chargedPFCand->muonRef()->globalTrack().isNonnull() ) track = chargedPFCand->muonRef()->globalTrack().get();
	else if ( chargedPFCand->muonRef().isNonnull() && chargedPFCand->muonRef()->outerTrack().isNonnull()  ) track = chargedPFCand->muonRef()->outerTrack().get();
	else if ( chargedPFCand->gsfTrackRef().isNonnull() ) track = chargedPFCand->gsfTrackRef().get();
      } 
      if ( nextChargedHadron->getTrack().isNonnull() && !track ) {
	track = nextChargedHadron->getTrack().get();
      }

      // discard candidate in case its track is "used" by any ChargedHadron in the clean collection
      bool isTrack_overlap = false;
      if ( track ) {
	double track_eta = track->eta();
	double track_phi = track->phi();
	for ( std::list<etaPhiPair>::const_iterator trackInCleanCollection = tracksInCleanCollection.begin();
	      trackInCleanCollection != tracksInCleanCollection.end(); ++trackInCleanCollection ) {
	  double dR = deltaR(track_eta, track_phi, trackInCleanCollection->first, trackInCleanCollection->second);
	  if ( dR < 1.e-4 ) isTrack_overlap = true;
	}
      }
      if ( verbosity_ ) {
	edm::LogPrint("PFRecoTauChHProducer")<< "isTrack_overlap = " << isTrack_overlap ;
      }
      if ( isTrack_overlap ) continue;

      // discard ChargedHadron candidates without track in case they are close to neutral PFCandidates "used" by ChargedHadron candidates in the clean collection
      bool isNeutralPFCand_overlap = false;
      if ( nextChargedHadron->algoIs(reco::PFRecoTauChargedHadron::kPFNeutralHadron) ) {
	for ( std::set<reco::PFCandidatePtr>::const_iterator neutralPFCandInCleanCollection = neutralPFCandsInCleanCollection.begin();
	      neutralPFCandInCleanCollection != neutralPFCandsInCleanCollection.end(); ++neutralPFCandInCleanCollection ) {
	  if ( (*neutralPFCandInCleanCollection) == nextChargedHadron->getChargedPFCandidate() ) isNeutralPFCand_overlap = true;
	}
      }
      if ( verbosity_ ) {
	edm::LogPrint("PFRecoTauChHProducer")<< "isNeutralPFCand_overlap = " << isNeutralPFCand_overlap ;
      }
      if ( isNeutralPFCand_overlap ) continue;
      
      // find neutral PFCandidates that are not "used" by any ChargedHadron in the clean collection
      std::vector<reco::PFCandidatePtr> uniqueNeutralPFCands;
      std::set_difference(nextChargedHadron->getNeutralPFCandidates().begin(),
			  nextChargedHadron->getNeutralPFCandidates().end(),
			  neutralPFCandsInCleanCollection.begin(),
			  neutralPFCandsInCleanCollection.end(),
			  std::back_inserter(uniqueNeutralPFCands));
      
      if ( uniqueNeutralPFCands.size() == nextChargedHadron->getNeutralPFCandidates().size() ) { // all neutral PFCandidates are unique, add ChargedHadron candidate to clean collection
	if ( track ) tracksInCleanCollection.push_back(std::make_pair(track->eta(), track->phi()));
	neutralPFCandsInCleanCollection.insert(nextChargedHadron->getNeutralPFCandidates().begin(), nextChargedHadron->getNeutralPFCandidates().end());
	if ( verbosity_ ) {
	  edm::LogPrint("PFRecoTauChHProducer")<< "--> adding nextChargedHadron to output collection." ;
	}
	cleanedChargedHadrons.push_back(*nextChargedHadron);
      } else { // remove overlapping neutral PFCandidates, reevaluate ranking criterion and process ChargedHadron candidate again
	nextChargedHadron->neutralPFCandidates_.clear();
    for(auto const& neutralPFCand : uniqueNeutralPFCands ) {
          nextChargedHadron->neutralPFCandidates_.push_back(neutralPFCand);
        }
	// update ChargedHadron four-momentum
	reco::tau::setChargedHadronP4(*nextChargedHadron);
	// reinsert ChargedHadron candidate into list of uncleaned candidates,
	// at position according to new rank
	ChargedHadronList::iterator insertionPoint = std::lower_bound(uncleanedChargedHadrons.begin(), uncleanedChargedHadrons.end(), *nextChargedHadron, *predicate_);
	if ( verbosity_ ) {
	  edm::LogPrint("PFRecoTauChHProducer")<< "--> removing non-unique neutral PFCandidates and reinserting nextChargedHadron in uncleaned collection." ;
	}
        uncleanedChargedHadrons.insert(insertionPoint, nextChargedHadron);
      }
    }

    if ( verbosity_ ) {
      print(cleanedChargedHadrons);
    }

    // add ChargedHadron-to-jet association
    pfJetChargedHadronAssociations->setValue(pfJet.key(), cleanedChargedHadrons);
  }

  evt.put(std::move(pfJetChargedHadronAssociations));
}

template <typename T>
void PFRecoTauChargedHadronProducer::print(const T& chargedHadrons) 
{
  for ( typename T::const_iterator chargedHadron = chargedHadrons.begin();
	chargedHadron != chargedHadrons.end(); ++chargedHadron ) {
    edm::LogPrint("PFRecoTauChHProducer") << (*chargedHadron);
    edm::LogPrint("PFRecoTauChHProducer") << "Rankers:" ;
    for ( rankerList::const_iterator ranker = rankers_.begin();
	  ranker != rankers_.end(); ++ranker) {
      const unsigned width = 25;
      edm::LogPrint("PFRecoTauChHProducer") << " " << std::setiosflags(std::ios::left) << std::setw(width) << ranker->name()
	     << " " << std::resetiosflags(std::ios::left) << std::setprecision(3) << (*ranker)(*chargedHadron) << std::endl;
    }
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(PFRecoTauChargedHadronProducer);
