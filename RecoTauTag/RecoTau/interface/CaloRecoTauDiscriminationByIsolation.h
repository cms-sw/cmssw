#ifndef RecoTauTag_RecoTau_CaloRecoTauDiscriminationByIsolation_H_
#define RecoTauTag_RecoTau_CaloRecoTauDiscriminationByIsolation_H_

/* class CaloRecoTauDiscriminationByIsolation
 * created : Jul 23 2007,
 * revised : Sep 5 2007,
 * contributors : Ludovic Houchu (Ludovic.Houchu@cern.ch ; IPHC, Strasbourg), Christian Veelken (veelken@fnal.gov ; UC Davis)
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TauReco/interface/CaloTau.h"
#include "DataFormats/TauReco/interface/CaloTauDiscriminatorByIsolation.h"

#include "RecoTauTag/TauTagTools/interface/CaloTauElementsOperators.h"

using namespace std; 
using namespace edm;
using namespace edm::eventsetup; 
using namespace reco;

class CaloRecoTauDiscriminationByIsolation : public EDProducer {
 public:
  explicit CaloRecoTauDiscriminationByIsolation(const ParameterSet& iConfig){   
    CaloTauProducer_                       = iConfig.getParameter<string>("CaloTauProducer");
    ApplyDiscriminationByTrackerIsolation_ = iConfig.getParameter<bool>("ApplyDiscriminationByTrackerIsolation");
    TrackerIsolAnnulus_Tracksmaxn_         = iConfig.getParameter<int>("TrackerIsolAnnulus_Tracksmaxn");   
    
    produces<CaloTauDiscriminatorByIsolation>();
  }
  ~CaloRecoTauDiscriminationByIsolation(){} 
  virtual void produce(Event&, const EventSetup&);
 private:  
  string CaloTauProducer_;
  bool ApplyDiscriminationByTrackerIsolation_;
  int TrackerIsolAnnulus_Tracksmaxn_;   
};
#endif
