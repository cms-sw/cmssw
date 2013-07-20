// -*- C++ -*-
//
// Package:    EventFilter/StorageManager
// Class:      SMHttpConsumerExample
// 
/**\class SMHttpConsumerExample

  Description: Example DQM Event Consumer Client 

  $Id: SMHttpConsumerExample.cc,v 1.2 2008/01/22 19:28:37 muzaffar Exp $

*/


// system include files
#include <memory>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"



using std::cout; using std::endl;

//
// class declaration
//

class SMHttpConsumerExample : public edm::EDAnalyzer {
public:
   explicit SMHttpConsumerExample( const edm::ParameterSet& );
   ~SMHttpConsumerExample();
   
  void analyze( const edm::Event&, const edm::EventSetup& );

  void endLuminosityBlock(edm::LuminosityBlock const& lumiSeg,
                          edm::EventSetup const& eSetup);
  void endRun(edm::Run const& run, edm::EventSetup const& eSetup);

  void endJob();

private:
  // event counter
  int counter;

};

SMHttpConsumerExample::SMHttpConsumerExample( const edm::ParameterSet& iConfig )
  : counter(0)
{
}


SMHttpConsumerExample::~SMHttpConsumerExample()
{
}

void SMHttpConsumerExample::endJob()
{
  std::cout << "Doing end of job processing after reading " 
            << counter << " events" << std::endl;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void SMHttpConsumerExample::analyze(const edm::Event& iEvent, 
			       const edm::EventSetup& iSetup )
{   
  ++counter;
}

void SMHttpConsumerExample::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, 
                                            edm::EventSetup const& eSetup)
{
  std::cout << "Doing end of lumi processing for lumi number "
            << lumiSeg.luminosityBlock() << " of run " 
            << lumiSeg.run() << std::endl;
}

void SMHttpConsumerExample::endRun(edm::Run const& run, edm::EventSetup const& eSetup)
{
  std::cout << "Doing end of run processing for run number "
            <<  run.run() << std::endl;
}

//define this as a plug-in
DEFINE_FWK_MODULE(SMHttpConsumerExample);
