#ifndef HLTDTActivityFilter_h
#define HLTDTActivityFilter_h
// -*- C++ -*-
//
// Package:    HLTDTActivityFilter
// Class:      HLTDTActivityFilter
// 
/**\class HLTDTActivityFilter HLTDTActivityFilter.cc filter/HLTDTActivityFilter/src/HLTDTActivityFilter.cc

Description: Filter to select HCAL abort gap events

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Carlo Battilana
//         Created:  Tue Jan 22 13:55:00 CET 2008
// $Id: HLTDTActivityFilter.h,v 1.1 2009/08/21 08:11:23 bdahmes Exp $
//
//


// include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"

#include <string>

//
// class declaration
//

class HLTDTActivityFilter : public HLTFilter {
public:
  explicit HLTDTActivityFilter(const edm::ParameterSet&);
  virtual ~HLTDTActivityFilter();
  
private:
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  
  // ----------member data ---------------------------

  /// input
  edm::InputTag inputDCC_ ; 
  edm::InputTag inputDDU_ ; 
  edm::InputTag inputDigis_ ; 

  bool processDCC_, processDDU_, processDigis_;
  int processingMode_;
  int minQual_;
  int maxStation_;
  int minBX_;
  int maxBX_;
  int minActiveChambs_;
  int minChambLayers_;

};

#endif
