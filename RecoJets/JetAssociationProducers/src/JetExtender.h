// \class JetExtender JetExtender.cc 
//
// Combines different Jet associations into single compact object
// which extends basic Jet information
// Fedor Ratnikov Sep. 10, 2007
// $Id: JetExtender.h,v 1.2 2010/02/17 17:50:18 wmtan Exp $
//
//
#ifndef JetExtender_h
#define JetExtender_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/Common/interface/EDProductfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class JetExtender : public edm::EDProducer {
   public:
      JetExtender(const edm::ParameterSet&);
      virtual ~JetExtender();

      virtual void produce(edm::Event&, const edm::EventSetup&);

   private:
     edm::InputTag mJets;
     edm::InputTag mJet2TracksAtVX;
     edm::InputTag mJet2TracksAtCALO;
};

#endif
