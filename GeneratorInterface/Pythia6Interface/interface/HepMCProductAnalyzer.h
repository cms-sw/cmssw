#ifndef HepMCProductAnalyzer_h
#define HepMCProductAnalyzer_h
// -*- C++ -*-
//
// Package:    HepMCProductAnalyzer
// Class:      HepMCProductAnalyzer
// 
/**\class HepMCProductAnalyzer HepMCProductAnalyzer.cc IOMC/HepMCProductAnalyzer/src/HepMCProductAnalyzer.cc

 Description: allows to print content of HepMCProducts

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Filip Moortgat
//         Created:  Mon Jan 16 17:56:07 CET 2006
// $Id: HepMCProductAnalyzer.h,v 1.1 2006/01/19 20:07:06 fmoortga Exp $
//
//


// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


//#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
//
// class decleration
//

class HepMCProductAnalyzer : public edm::EDAnalyzer {
   public:
      explicit HepMCProductAnalyzer(const edm::ParameterSet&);
      ~HepMCProductAnalyzer();


      virtual void analyze(const edm::Event&, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

 std::string label_;

};
#endif
