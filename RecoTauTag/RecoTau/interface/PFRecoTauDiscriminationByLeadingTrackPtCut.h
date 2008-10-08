#ifndef RecoTauTag_RecoTau_PFRecoTauDiscriminationByLeadingTrackPtCut_H_
#define RecoTauTag_RecoTau_PFRecoTauDiscriminationByLeadingTrackPtCut_H_

/* class PFRecoTauDiscriminationByLeadingTrackPtCut
 * created : October 08 2008,
 * revised : ,
 * Authorss : Simone Gennai (SNS)
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/TauReco/interface/PFTauDiscriminator.h"
#include "DataFormats/TrackReco/interface/Track.h"


using namespace std; 
using namespace edm;
//using namespace edm::eventsetup;
using namespace reco;

class PFRecoTauDiscriminationByLeadingTrackPtCut : public EDProducer {
 public:
  explicit PFRecoTauDiscriminationByLeadingTrackPtCut(const ParameterSet& iConfig){   
    PFTauProducer_        = iConfig.getParameter<InputTag>("PFTauProducer");
    minPtLeadTrack_ = iConfig.getParameter<double>("MinPtLeadingTrack");
    produces<PFTauDiscriminator>();
  }
  ~PFRecoTauDiscriminationByLeadingTrackPtCut(){} 
  virtual void produce(Event&, const EventSetup&);
 private:
  InputTag PFTauProducer_;
  double minPtLeadTrack_;

};
#endif
