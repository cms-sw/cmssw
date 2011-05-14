#ifndef HLTMhtProducer_h
#define HLTMhtProducer_h

/** \class HLTMhtProducer
 *
 *  \author Gheorghe Lungu
 *
 */

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

//
// class declaration
//

class HLTMhtProducer : public edm::EDProducer {

   public:
      explicit HLTMhtProducer(const edm::ParameterSet&);
      ~HLTMhtProducer();
      static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
      virtual void produce(edm::Event&, const edm::EventSetup&);
      
 private:
      edm::InputTag inputJetTag_; // input tag identifying jets
      double minPtJet_;
      double etaJet_;
      bool usePt_;
      
};

#endif //HLTMhtProducer_h
