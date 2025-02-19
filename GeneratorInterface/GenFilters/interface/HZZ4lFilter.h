#ifndef HZZ4lFilter_H
#define HZZ4lFilter_H

// -*- C++ -*-
//
// Package:    HZZ4lFilter
// Class:      HZZ4lFilter
// 
/**\class HZZ4lFilter HZZ4lFilter.cc IOMC/HZZ4lFilter/src/HZZ4lFilter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Puljak Ivica
//         Created:  Wed Apr 18 12:52:31 CEST 2007
// $Id: HZZ4lFilter.h,v 1.2 2009/12/15 10:29:32 fabiocos Exp $
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
// class declaration
//

class HZZ4lFilter : public edm::EDFilter {
   public:
      explicit HZZ4lFilter(const edm::ParameterSet&);
      ~HZZ4lFilter();

   private:
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      // virtual void endJob() ;
      
      // ----------member data ---------------------------

       std::string label_;
       
       double minPtElectronMuon;
       double maxEtaElectronMuon;

};

#endif
