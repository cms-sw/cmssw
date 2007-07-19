// -*- C++ -*-
//
// Package:    GctFibreAnalyzer
// Class:      GctFibreAnalyzer
// 
/**\class GctFibreAnalyzer GctFibreAnalyzer.cc L1Trigger/L1GctAnalzyer/src/GctFibreAnalyzer.cc

Description: Analyzer individual fibre channels from the source card.

*/
//
// Original Author:  Alex Tapper
//         Created:  Thu Jul 12 14:21:06 CEST 2007
// $Id: GctFibreAnalyzer.cc,v 1.1 2007/07/18 13:14:07 tapper Exp $
//
//

// Include file
#include "L1Trigger/L1GctAnalyzer/interface/GctFibreAnalyzer.h"

// Data format
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

GctFibreAnalyzer::GctFibreAnalyzer(const edm::ParameterSet& iConfig):
  m_fibreSource(iConfig.getUntrackedParameter<edm::InputTag>("FibreSource"))
{
}


GctFibreAnalyzer::~GctFibreAnalyzer()
{
}

void GctFibreAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  
  Handle<L1GctFibreCollection> fibre;
  iEvent.getByLabel(m_fibreSource,fibre);

  bool bc0= false;

  for (L1GctFibreCollection::const_iterator f=fibre->begin(); f!=fibre->end(); f++){

    // This is where we'll check it the logical ID corresponds to what we expect
    // logical id pattern= (logicalID<<8) | (sfpNo+1) for sfpNo=0,1,2,3
    // source card type = 0 for electron cards
    // source card local ID = source card type (+4 if RCt crate >=9)
    // source card logical ID = 8*(RCT phi region 0-8) + source card local ID

    // For now write them out
    std::cout << std::hex
              << "data=" << f->data() 
              << " block=" << f->block() 
              << " index=" << f->index() 
              << " bx=" << f->bx()
              << std::endl;

    // Check that the last bit on cycle 1 is set as it should be
    if (!(f->data() & 0x80000000)){
      std::cout << "Error in fibre data, b15 on cycle 1 not set! Data corrupt." << std::endl;
      std::cout << std::hex
                << "data=" << f->data() 
                << " block=" << f->block() 
                << " index=" << f->index() 
                << " bx=" << f->bx()
                << std::endl;
    }

    // Check for BC0 on this event
    if ((f->data() & 0x8000) && (f==fibre->begin())) bc0=true;

    // If this event has a BC0 then check this fibre sees it too
    if (bc0 && !(f->data() & 0x8000)){
      std::cout << "Error in fibre data, missing BC0! Data corrupt." << std::endl;
      std::cout << std::hex
                << "data=" << f->data() 
                << " block=" << f->block() 
                << " index=" << f->index() 
                << " bx=" << f->bx()
                << std::endl;
    }
  }   
}


void GctFibreAnalyzer::beginJob(const edm::EventSetup&)
{
}

void GctFibreAnalyzer::endJob() 
{
}

