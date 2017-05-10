#ifndef PYTHIAMOMDAUFILTER_h
#define PYTHIAMOMDAUFILTER_h
// -*- C++ -*-
//
// Package:    PythiaMomDauFilter
// Class:      PythiaMomDauFilter
// 
/**\class PythiaMomDauFilter PythiaMomDauFilter.cc 

 Description: Filter events using MotherId and ChildrenIds infos

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Daniele Pedrini
//         Created:  Oct 27 2015
// Fixed : Ta-Wei Wang, Dec 11 2015        
// $Id: PythiaMomDauFilter.h,v 1.1 2015/10/27  pedrini Exp $
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

class PythiaMomDauFilter : public edm::EDFilter {
   public:
      explicit PythiaMomDauFilter(const edm::ParameterSet&);
      ~PythiaMomDauFilter();


      virtual bool filter(edm::Event&, const edm::EventSetup&);
   private:
      // ----------memeber function----------------------

      // ----------member data ---------------------------
      
       std::string label_;
       std::vector<int> dauIDs;
       std::vector<int> desIDs;
       int particleID;
       int daughterID;
       bool chargeconju; 
       int ndaughters;
       int ndescendants;
       double minptcut;
       double maxptcut;
       double minetacut;
       double maxetacut;
       double mom_minptcut;
       double mom_maxptcut;
       double mom_minetacut;
       double mom_maxetacut;
};
#define PYCOMP pycomp_
extern "C" {
 int PYCOMP(int& ip);
} 
#endif
DEFINE_FWK_MODULE(PythiaMomDauFilter);
