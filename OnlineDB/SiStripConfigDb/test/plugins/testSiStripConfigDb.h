// Last commit: $Id: testSiStripConfigDb.h,v 1.3 2010/01/07 11:26:05 lowette Exp $
// Latest tag:  $Name: CMSSW_6_2_0 $
// Location:    $Source: /local/reps/CMSSW/CMSSW/OnlineDB/SiStripConfigDb/test/plugins/testSiStripConfigDb.h,v $

#ifndef OnlineDB_SiStripConfigDb_testSiStripConfigDb_H
#define OnlineDB_SiStripConfigDb_testSiStripConfigDb_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class SiStripConfigDb;

/**
   @class testSiStripConfigDb 
   @author R.Bainbridge
   @brief Simple class that tests SiStripConfigDb service
*/
class testSiStripConfigDb : public edm::EDAnalyzer {

 public:
  
  testSiStripConfigDb( const edm::ParameterSet& );
  ~testSiStripConfigDb();
  
  void beginJob();
  void analyze( const edm::Event&, const edm::EventSetup& ) {;}
  void endJob() {;}

 private:

  SiStripConfigDb* db_;

  bool download_;

  bool upload_;

  bool conns_;

  bool devices_;

  bool feds_;

  bool dcus_;

  bool anals_;
  
};

#endif // OnlineDB_SiStripConfigDb_testSiStripConfigDb_H

