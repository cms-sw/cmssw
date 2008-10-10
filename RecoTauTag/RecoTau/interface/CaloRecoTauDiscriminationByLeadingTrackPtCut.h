#ifndef RecoTauTag_RecoTau_CaloRecoTauDiscriminationByLeadingTrackPtCut_H_
#define RecoTauTag_RecoTau_CaloRecoTauDiscriminationByLeadingTrackPtCut_H_

/* class CaloRecoTauDiscriminationByLeadingTrackPtCut
 * created : October 08 2008,
 * revised : ,
 * Authorss : Simone Gennai (SNS)
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauDiscriminatorByIsolation.h"
#include "DataFormats/TrackReco/interface/Track.h"


using namespace std; 
using namespace edm;
//using namespace edm::eventsetup;
using namespace reco;

class CaloRecoTauDiscriminationByLeadingTrackPtCut : public EDProducer {
 public:
  explicit CaloRecoTauDiscriminationByLeadingTrackPtCut(const ParameterSet& iConfig){   
    CaloTauProducer_        = iConfig.getParameter<InputTag>("CaloTauProducer");
    minPtLeadTrack_ = iConfig.getParameter<double>("MinPtLeadingTrack");
    produces<CaloTauDiscriminatorByIsolation>();
  }
  ~CaloRecoTauDiscriminationByLeadingTrackPtCut(){} 
  virtual void produce(Event&, const EventSetup&);
 private:
  InputTag CaloTauProducer_;
  double minPtLeadTrack_;

};
#endif
