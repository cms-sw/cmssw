// Last commit: $Id: testSiStripGainBuilderFromDb.h,v 1.1 2008/09/29 13:20:52 bainbrid Exp $

#ifndef OnlineDB_SiStripESSources_testSiStripGainBuilderFromDb_H
#define OnlineDB_SiStripESSources_testSiStripGainBuilderFromDb_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**
   @class testSiStripGainBuilderFromDb 
   @brief Analyzes FEC (and FED) cabling object(s)
*/
class testSiStripGainBuilderFromDb : public edm::EDAnalyzer {

 public:
  
  testSiStripGainBuilderFromDb( const edm::ParameterSet& ) {;}
  virtual ~testSiStripGainBuilderFromDb() {;}
  
  void beginRun( const edm::Run&, const edm::EventSetup& );
  void analyze( const edm::Event&, const edm::EventSetup& ) {;}
  
};

#endif // OnlineDB_SiStripESSources_testSiStripGainBuilderFromDb_H
