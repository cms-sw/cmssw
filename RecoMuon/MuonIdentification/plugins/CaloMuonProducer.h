// -*- C++ -*-
//
// Package:    CaloMuonProducer
// Class:      CaloMuonProducer
//
// Original Author:  Dmytro Kovalskyi
//         Created:  Wed Oct  3 16:29:03 CDT 2007
// $Id: CaloMuonProducer.h,v 1.3 2009/09/23 19:15:04 dmytro Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

class CaloMuonProducer : public edm::EDProducer {
 public:
   explicit CaloMuonProducer(const edm::ParameterSet&);
   ~CaloMuonProducer();
   
 private:
   virtual void     produce( edm::Event&, const edm::EventSetup& );
   edm::InputTag inputCollection;
};
