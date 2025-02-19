// -*- C++ -*-
//
// Package:    CalibTracker/SiStripDCS/plugins
// Class:      SyncDCSO2O
// 
/**\class FilterTrackerOn FilterTrackerOn.cc

 Description: EDFilter returning true when the number of modules with HV on in the Tracker is
              above a given threshold.

*/
//
// Original Author:  Marco DE MATTIA
//         Created:  2010/03/10 10:51:00
// $Id: FilterTrackerOn.h,v 1.2 2010/03/29 12:32:37 demattia Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

class FilterTrackerOn : public edm::EDFilter
{
 public:
  explicit FilterTrackerOn(const edm::ParameterSet&);
  ~FilterTrackerOn();

 private:
  virtual void beginJob() ;
  virtual bool filter(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  int minModulesWithHVoff_;
};
