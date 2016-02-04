#ifndef PythiaHLTSoupFilter_h
#define PythiaHLTSoupFilter_h
// -*- C++ -*-
//
// Package:    PythiaHLTSoupFilter
// Class:      PythiaHLTSoupFilter
// 
/**\class PythiaHLTSoupFilter PythiaHLTSoupFilter.cc IOMC/PythiaHLTSoupFilter/src/PythiaHLTSoupFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Filip Moortgat
//         Created:  Mon Jan 23 14:57:54 CET 2006
// $Id: PythiaHLTSoupFilter.h,v 1.2 2010/07/21 04:23:25 wmtan Exp $
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

class PythiaHLTSoupFilter : public edm::EDFilter {
   public:
      explicit PythiaHLTSoupFilter(const edm::ParameterSet&);
      ~PythiaHLTSoupFilter();


      virtual bool filter(edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------
      
       std::string label_;
       
       double minptelectron;
       double minptmuon;
       double maxetaelectron;
       double maxetamuon;
       double minpttau;
       double maxetatau;
       
         
};
#endif
