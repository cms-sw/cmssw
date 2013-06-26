#ifndef HLTDTActivityFilter_h
#define HLTDTActivityFilter_h
// -*- C++ -*-
//
// Package:    HLTDTActivityFilter
// Class:      HLTDTActivityFilter
// 


/*

Description: Filter to select events with activity in the muon barrel system

*/


//
// Original Author:  Carlo Battilana
//         Created:  Tue Jan 22 13:55:00 CET 2008
// $Id: HLTDTActivityFilter.h,v 1.4 2012/01/21 15:00:13 fwyzard Exp $
//
//


// Fwk header files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"


// c++ header files
#include<bitset>
#include <string>

class DTGeometry;
class L1MuRegionalCand;

//
// class declaration
//

class HLTDTActivityFilter : public HLTFilter {
public:

  explicit HLTDTActivityFilter(const edm::ParameterSet&);
  virtual ~HLTDTActivityFilter();

private:

  virtual bool hltFilter(edm::Event&, const edm::EventSetup&, trigger::TriggerFilterObjectWithRefs & filterproduct);
  virtual bool beginRun(edm::Run& iRun, const edm::EventSetup& iSetup);

  bool hasActivity(const std::bitset<4> &);  
  bool matchChamber(const uint32_t &, const L1MuRegionalCand&);
  
  enum activityType { DCC=0, DDU=1, RPC=2, DIGI=3 };
  

  // ----------member data ---------------------------

  edm::InputTag inputTag_[4]; 
  bool process_[4];
  std::bitset<15> activeSecs_;

  edm::ESHandle<DTGeometry> dtGeom_;

  bool  orTPG_;
  bool  orRPC_;
  bool  orDigi_;

  int   minQual_;
  int   maxStation_;
  int   minBX_[3];
  int   maxBX_[3];
  int   minActiveChambs_;
  int   minChambLayers_;
  
  float maxDeltaPhi_;
  float maxDeltaEta_;

};

#endif
