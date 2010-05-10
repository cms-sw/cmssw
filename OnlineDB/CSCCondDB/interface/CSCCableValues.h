#ifndef _CSCCABLEVALUES_H
#define _CSCCABLEVALUES_H

#include <memory>
#include "FWCore/Framework/interface/SourceFactory.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include "CondFormats/CSCObjects/interface/CSCCables.h"
#include "CondFormats/DataRecord/interface/CSCCablesRcd.h"
#include "OnlineDB/CSCCondDB/interface/CSCCableValues.h"
#include "OnlineDB/CSCCondDB/interface/CSCCableRead.h"

class CSCCableValues: public edm::ESProducer, public edm::EventSetupRecordIntervalFinder  {
 public:
  CSCCableValues(const edm::ParameterSet&);
  ~CSCCableValues();
  
  inline static CSCCables * fillCables();

  typedef const  CSCCables * ReturnType;
  
  ReturnType produceCables(const CSCCablesRcd&);
  
 private:
  // ----------member data ---------------------------
  void setIntervalFor(const edm::eventsetup::EventSetupRecordKey &, const edm::IOVSyncValue&, edm::ValidityInterval & );
  CSCCables *cableObj ;

};

#include<fstream>
#include<vector>
#include<iostream>

// to workaround plugin library
inline CSCCables *  CSCCableValues::fillCables()
{
  CSCCables * cableobj = new CSCCables();
  csccableread *cable = new csccableread ();

  int i; //i - chamber index.
  int count=0;
  std::string chamber_label, cfeb_rev, alct_rev;
  int cfeb_length=0, alct_length=0, cfeb_tmb_skew_delay=0, cfeb_timing_corr=0;

  /* This is version for 481 chambers. */

  cableobj->cables.resize(481);
  for(i=1;i<=481;++i){
    cable->cable_read(i, &chamber_label, &cfeb_length, &cfeb_rev, &alct_length,
    &alct_rev, &cfeb_tmb_skew_delay, &cfeb_timing_corr);
    cableobj->cables[i-1].chamber_label=chamber_label;
    cableobj->cables[i-1].cfeb_length=cfeb_length;
    cableobj->cables[i-1].cfeb_rev=cfeb_rev;
    cableobj->cables[i-1].alct_length=alct_length;
    cableobj->cables[i-1].alct_rev=alct_rev;
    cableobj->cables[i-1].cfeb_tmb_skew_delay=cfeb_tmb_skew_delay;
    cableobj->cables[i-1].cfeb_timing_corr=cfeb_timing_corr;
    count=count+1;
  }
  return cableobj;
}
  

#endif
