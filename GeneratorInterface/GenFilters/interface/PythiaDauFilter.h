#ifndef PYTHIADAUFILTER_h
#define PYTHIADAUFILTER_h
// -*- C++ -*-
//
// Package:    PythiaDauFilter
// Class:      PythiaDauFilter
// 
/**\class PythiaDauFilter PythiaDauFilter.cc 

 Description: Filter events using MotherId and ChildrenIds infos

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Daniele Pedrini
//         Created:  Apr 29 2008
// $Id: PythiaDauFilter.h,v 1.2 2010/07/21 04:23:24 wmtan Exp $
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

class PythiaDauFilter : public edm::EDFilter {
   public:
      explicit PythiaDauFilter(const edm::ParameterSet&);
      ~PythiaDauFilter();


      virtual bool filter(edm::Event&, const edm::EventSetup&);
   private:
      // ----------memeber function----------------------

      // ----------member data ---------------------------
      
       std::string label_;
       std::vector<int> dauIDs;
       int particleID;
       bool chargeconju; 
       int ndaughters;
       double minptcut;
       double maxptcut;
       double minetacut;
       double maxetacut;
};
#define PYCOMP pycomp_
extern "C" {
 int PYCOMP(int& ip);
} 
#endif
