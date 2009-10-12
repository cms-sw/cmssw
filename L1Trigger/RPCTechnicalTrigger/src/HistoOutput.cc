// $Id: $
// Include files 



// local
#include "HistoOutput.h"

//-----------------------------------------------------------------------------
// Implementation file for class : HistoOutput
//
// 2009-02-05 : Andres Felipe Osorio Oliveros
//-----------------------------------------------------------------------------

//=============================================================================
// Standard constructor, initializes variables
//=============================================================================
HistoOutput::HistoOutput( edm::Service<TFileService> & fservice,  const char *opt ) {
  
  m_option = std::string(opt);
  
  hcompflag    = fservice->make<TH1D>("ComparisonFlag", 
                                      "Comparison flag between TTU/L1Trigger and Emulator", 
                                      13 , 0 , 12 );
  
  hnumGMTmuons = fservice->make<TH1D>("NumGMTMuons",
                                      "Number of GMT muon candidates per event",
                                      50 , 0 , 10 );
  
  htrigdecision = fservice->make<TH1D>("GMTFilterFlag", 
                                       "Distribution of GMT trigger decision flags over all events", 
                                       21 , 0 , 20 );
  
  hhtrgperevent = fservice->make<TH2D>("GMTFilterPerEvent", 
                                       "Distribution of GMT trigger decision flags per event/candidate", 
                                       101,0,100, 21, 0 , 20 );
  
  
}
//=============================================================================
// Destructor
//=============================================================================
HistoOutput::~HistoOutput() {} 

//=============================================================================
