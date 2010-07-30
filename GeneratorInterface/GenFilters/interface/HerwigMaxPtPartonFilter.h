#ifndef HERWIGMAXPTPARTONFILTER_h
#define HERWIGMAXPTPARTONFILTER_h
// -*- C++ -*-
//
// Package:    HerwigMaxPtPartonFilter
// Class:      HerwigMaxPtPartonFilter
// 
/**\class HerwigMaxPtPartonFilter HerwigMaxPtPartonFilter.cc IOMC/HerwigMaxPtPartonFilter/src/HerwigMaxPtPartonFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Brian Dorney
//         Created:  July 27th 2010
// $Id: HerwigMaxPtPartonFilter.h v1.0
//
// Modified From: PythiaFilter.cc
//
// Special Thanks to Filip Moortgat
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
// class decleration
//

class HerwigMaxPtPartonFilter : public edm::EDFilter {
public:
  explicit HerwigMaxPtPartonFilter(const edm::ParameterSet&);
  ~HerwigMaxPtPartonFilter();
  
  
  virtual bool filter(edm::Event&, const edm::EventSetup&);
private:
  // ----------member data ---------------------------
  
  std::string label_;
  
  bool accepted;
  bool isParton;
  
  double minptcut;
  double maxptcut;
  double maxPartonPt;
  
  int processID;
  int ChosenPartonId, ChosenPartonSt;
  int pos1stCluster;
  
  long counter;
};

#endif
