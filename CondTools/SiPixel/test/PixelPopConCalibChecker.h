#ifndef CondTools_SiPixel_PixelPopConCalibChecker_H
#define CondTools_SiPixel_PixelPopConCalibChecker_H

// -*- C++ -*-
//
// Package:    PixelPopConCalibChecker
// Class:      PixelPopConCalibChecker
// 
/**\class PixelPopConCalibChecker PixelPopConCalibChecker.h SiPixel/test/PixelPopConCalibChecker.h

 Description: Test analyzer for checking calib configuration objects written to db

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  M. Eads
//         Created:  August 2008
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

//
// class decleration
//
class PixelPopConCalibChecker : public edm::EDAnalyzer {
   public:
      explicit PixelPopConCalibChecker(const edm::ParameterSet&);
      ~PixelPopConCalibChecker();


   private:
      //virtual void beginJob(const edm::EventSetup&) ;
      virtual void beginJob();
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      std::string _filename;
      int _messageLevel;

      // ----------member data ---------------------------
};



#endif
