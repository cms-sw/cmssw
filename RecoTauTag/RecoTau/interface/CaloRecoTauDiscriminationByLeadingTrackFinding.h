#ifndef RecoTauTag_RecoTau_CaloRecoTauDiscriminationByLeadingTrackFinding_H_
#define RecoTauTag_RecoTau_CaloRecoTauDiscriminationByLeadingTrackFinding_H_

/* class CaloRecoTauDiscriminationByLeadingTrackFinding
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

class CaloRecoTauDiscriminationByLeadingTrackFinding : public EDProducer {
 public:
  explicit CaloRecoTauDiscriminationByLeadingTrackFinding(const ParameterSet& iConfig){   
    CaloTauProducer_        = iConfig.getParameter<InputTag>("CaloTauProducer");
    //    matchingCone__ = iConfig.getParameter<double>("LeadingTrackMatchingCone");
    produces<CaloTauDiscriminatorByIsolation>();
  }
  ~CaloRecoTauDiscriminationByLeadingTrackFinding(){} 
  virtual void produce(Event&, const EventSetup&);
 private:
  InputTag CaloTauProducer_;
  //  double matchingCone_;

};
#endif
