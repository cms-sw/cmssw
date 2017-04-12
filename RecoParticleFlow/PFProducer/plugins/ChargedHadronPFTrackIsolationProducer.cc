/*
 * ChargedHadronPFTrackIsolationProducer
 *
 * Author: Andreas Hinzmann
 *
 * Associates PF isolation flag to charged hadron candidates
 *
 */

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"

namespace edm {
   class ConfigurationDescriptions;
}

class ChargedHadronPFTrackIsolationProducer : public edm::stream::EDProducer<> 
{
public:

  explicit ChargedHadronPFTrackIsolationProducer(const edm::ParameterSet& cfg);
  ~ChargedHadronPFTrackIsolationProducer() {}
  void produce(edm::Event& evt, const edm::EventSetup& es);
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);

private:
  // input collection
  edm::InputTag srcCandidates_;
  edm::EDGetTokenT<edm::View<reco::PFCandidate> > Candidates_token;
  double minTrackPt_;
  double minRawCaloEnergy_;

};

ChargedHadronPFTrackIsolationProducer::ChargedHadronPFTrackIsolationProducer(const edm::ParameterSet& cfg)
{
  srcCandidates_ = cfg.getParameter<edm::InputTag>("src");
  Candidates_token = consumes<edm::View<reco::PFCandidate> >(srcCandidates_);
  minTrackPt_ = cfg.getParameter<double>("minTrackPt");
  minRawCaloEnergy_ = cfg.getParameter<double>("minRawCaloEnergy");
  
  produces<edm::ValueMap<bool> >();
}

void ChargedHadronPFTrackIsolationProducer::produce(edm::Event& evt, const edm::EventSetup& es) 
{
  // get a view of our Candidates via the base candidates
  typedef edm::View<reco::PFCandidate> PFCandidateView;
  edm::Handle<PFCandidateView> Candidates;
  evt.getByToken(Candidates_token, Candidates);
  
  std::vector<bool> values;

  for( PFCandidateView::const_iterator c = Candidates->begin(); c!=Candidates->end(); ++c) {
    // Check that there is only one track in the block.
    unsigned int nTracks = 0;
    if ((c->particleId()==1) && (c->pt()>minTrackPt_) && ((c->rawEcalEnergy()+c->rawHcalEnergy())>minRawCaloEnergy_)) {
      const reco::PFCandidate::ElementsInBlocks& theElements = c->elementsInBlocks();
      if( theElements.empty() ) continue;
      const reco::PFBlockRef blockRef = theElements[0].first;
      const edm::OwnVector<reco::PFBlockElement>& elements = blockRef->elements();
      for(unsigned int iEle=0; iEle<elements.size(); iEle++) {	         	 // Find the tracks in the block
         reco::PFBlockElement::Type type = elements[iEle].type();
         if( type== reco::PFBlockElement::TRACK)
           nTracks++;
      }
    }
    values.push_back((nTracks==1));
  }

  std::unique_ptr<edm::ValueMap<bool> > out(new edm::ValueMap<bool>());
  edm::ValueMap<bool>::Filler filler(*out);
  filler.insert(Candidates,values.begin(),values.end());
  filler.fill();
  evt.put(std::move(out));
}

void ChargedHadronPFTrackIsolationProducer::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
   edm::ParameterSetDescription desc;
   desc.add<edm::InputTag>("src", edm::InputTag("particleFlow"));
   desc.add<double>("minTrackPt", 1);
   desc.add<double>("minRawCaloEnergy", 0.5);
   descriptions.add("chargedHadronPFTrackIsolation", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(ChargedHadronPFTrackIsolationProducer);
