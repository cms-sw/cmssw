#ifndef JetMETCorrections_JetPlusTrack_RawPATJetProducer_h
#define JetMETCorrections_JetPlusTrack_RawPATJetProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSet;
}

class RawPATJetProducer : public edm::EDProducer {

 public:
  
  explicit RawPATJetProducer( const edm::ParameterSet& );
  ~RawPATJetProducer();

  virtual void produce( edm::Event&, const edm::EventSetup& );
  
 private:
  
  edm::InputTag jets_;
  
};

#endif // JetMETCorrections_JetPlusTrack_RawPATJetProducer_h
