// -*- C++ -*-
//
// Package:    AnaObjAnalyzer
// Class:      AnaObjAnalyzer
// 
/**\class AnaObjAnalyzer AnaObjAnalyzer.cc AnalysisExamples/AnaObjAnalyzer/src/AnaObjAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Marco De Mattia
//         Created:  Tue May  8 13:05:37 CEST 2007
// $Id: AnaObjAnalyzer.h,v 1.2 2007/07/20 17:26:45 demattia Exp $
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
// class declaration
//

class AnaObjAnalyzer : public edm::EDAnalyzer {
   public:
      explicit AnaObjAnalyzer(const edm::ParameterSet&);
      ~AnaObjAnalyzer();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
};
