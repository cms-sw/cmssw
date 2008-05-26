// Last commit: $Id: test_PedestalsBuilder.h,v 1.2 2007/03/28 10:30:14 bainbrid Exp $
// Latest tag:  $Name: V02-00-02 $
// Location:    $Source: /cvs_server/repositories/CMSSW/CMSSW/OnlineDB/SiStripESSources/test/stubs/test_PedestalsBuilder.h,v $

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
  
  void beginJob( edm::EventSetup const& );
  void analyze( const edm::Event&, const edm::EventSetup& ) {;}
  void endJob() {;}
  
};

#endif // OnlineDB_SiStripESSources_test_PedestalsBuilder_H

