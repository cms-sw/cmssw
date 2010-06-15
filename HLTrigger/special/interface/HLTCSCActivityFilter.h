#ifndef HLTCSCActivityFilter_h
#define HLTCSCActivityFilter_h
// -*- C++ -*-
//
// Package:    HLTCSCActivityFilter
// Class:      HLTCSCActivityFilter
// 
/**\class HLTCSCActivityFilter HLTCSCActivityFilter.cc filter/HLTCSCActivityFilter/src/HLTCSCActivityFilter.cc

Description: Filter to select HCAL abort gap events

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Carlo Battilana
//         Created:  Tue Jan 22 13:55:00 CET 2008
// $Id: HLTCSCActivityFilter.h,v 1.2 2010/03/08 10:54:32 goys Exp $
//
//


// include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include "DataFormats/CSCDigi/interface/CSCStripDigi.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include<bitset>
#include <string>

//
// class declaration
//

class HLTCSCActivityFilter : public HLTFilter {
public:
  explicit HLTCSCActivityFilter(const edm::ParameterSet&);
  virtual ~HLTCSCActivityFilter();
  
private:
  virtual bool filter(edm::Event&, const edm::EventSetup&);
    bool applyfilter;
  // ----------member data ---------------------------

  /// input
    edm::InputTag m_cscStripDigiTag;
    
    bool processDigis_;
    bool MESR;
    int RingNumb;
    int StationNumb;
    
};

#endif
