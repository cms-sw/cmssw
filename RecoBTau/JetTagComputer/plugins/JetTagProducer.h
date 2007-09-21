#ifndef RecoBTag_JetTagProducer
#define RecoBTag_JetTagProducer

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"

class JetTagProducer : public edm::EDProducer {
   public:
      explicit JetTagProducer(const edm::ParameterSet&);
      ~JetTagProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&);
   private:
    const edm::ParameterSet& m_config;
    edm::InputTag m_tagInfo;
    const JetTagComputer * m_computer;
    std::string m_jetTagComputer;
};
#endif
