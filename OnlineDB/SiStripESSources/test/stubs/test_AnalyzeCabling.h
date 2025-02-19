// Last commit: $Id: test_AnalyzeCabling.h,v 1.2 2008/06/05 14:59:15 bainbrid Exp $

#ifndef OnlineDB_SiStripESSources_test_AnalyzeCabling_H
#define OnlineDB_SiStripESSources_test_AnalyzeCabling_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**
   @class test_AnalyzeCabling 
   @brief Analyzes FEC (and FED) cabling object(s)
*/
class test_AnalyzeCabling : public edm::EDAnalyzer {

 public:
  
  test_AnalyzeCabling( const edm::ParameterSet& ) {;}
  virtual ~test_AnalyzeCabling() {;}
  
  void beginRun( const edm::Run&, const edm::EventSetup& );
  void analyze( const edm::Event&, const edm::EventSetup& ) {;}
  
};

#endif // OnlineDB_SiStripESSources_test_AnalyzeCabling_H
