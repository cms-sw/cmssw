#ifndef PYTHIADAUVFILTER_h
#define PYTHIADAUVFILTER_h
// -*- C++ -*-
//
// Package:    PythiaDauVFilter
// Class:      PythiaDauVFilter
// 
/**\class PythiaDauVFilter PythiaDauVFilter.cc 

 Description: Filter events using MotherId and ChildrenIds infos

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Daniele Pedrini
//         Created:  Apr 29 2008
// $Id: PythiaDauVFilter.h,v 1.1 2012/02/08 20:10:00 lenzip Exp $
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


//
// class decleration
//

class PythiaDauVFilter : public edm::EDFilter {
 public:
  explicit PythiaDauVFilter(const edm::ParameterSet&);
  ~PythiaDauVFilter();
  
  
  virtual bool filter(edm::Event&, const edm::EventSetup&);
 private:
  int fVerbose;  
  std::string label_;
  std::vector<int> dauIDs;
  int particleID;
  int motherID;
  bool chargeconju; 
  int ndaughters;
  std::vector<double> minptcut;
  double maxptcut;
  std::vector<double> minetacut;
  std::vector<double> maxetacut;
};
#define PYCOMP pycomp_
extern "C" {
  int PYCOMP(int& ip);
} 
#endif
