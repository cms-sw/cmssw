// Last commit: $Id: test_PedestalsBuilder.h,v 1.5 2013/05/30 21:52:09 gartung Exp $
// Latest tag:  $Name: CMSSW_6_2_0 $
// Location:    $Source: /local/reps/CMSSW/CMSSW/OnlineDB/SiStripESSources/test/stubs/test_PedestalsBuilder.h,v $

#ifndef OnlineDB_SiStripESSources_test_PedestalsBuilder_H
#define OnlineDB_SiStripESSources_test_PedestalsBuilder_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**
   @class test_PedestalsBuilder 
   @brief Simple class that analyzes Digis produced by RawToDigi unpacker
*/
class test_PedestalsBuilder : public edm::EDAnalyzer {

 public:
  
  test_PedestalsBuilder( const edm::ParameterSet& ) {;}
  virtual ~test_PedestalsBuilder() {;}
  
  void beginJob(){;}
  void analyze( const edm::Event&, const edm::EventSetup& );
  void endJob() {;}
  
};

#endif // OnlineDB_SiStripESSources_test_PedestalsBuilder_H

