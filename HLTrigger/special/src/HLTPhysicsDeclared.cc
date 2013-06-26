// -*- C++ -*-
//
// Package:   HLTPhysicsDeclared
// Class:     HLTPhysicsDeclared
//
// Original Author:     Luca Malgeri
// Adapted for HLT by:  Andrea Bocci

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
//
// class declaration
//

class HLTPhysicsDeclared : public edm::EDFilter {
public:
  explicit HLTPhysicsDeclared( const edm::ParameterSet & );
  ~HLTPhysicsDeclared();
  
private:
  virtual bool filter( edm::Event &, const edm::EventSetup & );

  bool          m_invert;
  edm::InputTag m_gtDigis;
  
};

// system include files
#include <memory>

// user include files
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GtFdlWord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"

using namespace edm;

HLTPhysicsDeclared::HLTPhysicsDeclared(const edm::ParameterSet & config) :
  m_invert(  config.getParameter<bool>("invert") ),
  m_gtDigis( config.getParameter<edm::InputTag>("L1GtReadoutRecordTag") )
{
}

HLTPhysicsDeclared::~HLTPhysicsDeclared()
{
}

bool HLTPhysicsDeclared::filter( edm::Event & event, const edm::EventSetup & setup)
{
  bool accept = false;

  if (event.isRealData()) {
    // for real data, access the "physics enabled" bit in the L1 GT data
    edm::Handle<L1GlobalTriggerReadoutRecord> h_gtDigis;
    if (not event.getByLabel(m_gtDigis, h_gtDigis)) {
      edm::LogWarning(h_gtDigis.whyFailed()->category()) << h_gtDigis.whyFailed()->what();
      // not enough informations to make a decision - reject the event
      return false;
    } else {
      L1GtFdlWord fdlWord = h_gtDigis->gtFdlWord();
      if (fdlWord.physicsDeclared() == 1) 
        accept = true;
    }
  } else {
    // for MC, assume the "physics enabled" bit to be always set
    accept = true;
  }

  // if requested, invert the filter decision
  if (m_invert)
    accept = not accept;

  return accept;
}

// define this as a framework plug-in
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTPhysicsDeclared);
