#ifndef MinimumBiasFilter_H
#define MinimumBiasFilter_H
//
// Original Author:  Filippo Ambroglini
//         Created:  Fri Sep 29 17:10:41 CEST 2006
// $Id: MinimumBiasFilter.h,v 1.1 2007/03/28 14:04:44 fabstoec Exp $
//
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
// class declaration
//

class MinimumBiasFilter : public edm::EDFilter {
 public:
  MinimumBiasFilter(const edm::ParameterSet&);
  virtual ~MinimumBiasFilter() {};
  
 private:
  bool filter(edm::Event&, const edm::EventSetup&);
  // ----------member data ---------------------------
  float theEventFraction;
  
};
#endif
