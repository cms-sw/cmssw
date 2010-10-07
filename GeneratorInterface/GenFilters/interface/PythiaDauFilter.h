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
// $Id: PythiaDauFilter.h,v 1.1 2008/04/29 14:42:28 pedrini Exp $
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
