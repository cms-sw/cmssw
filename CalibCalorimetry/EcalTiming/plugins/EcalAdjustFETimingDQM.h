// -*- C++ -*-
//
// Package:    EcalTiming
// Class:      EcalAdjustFETimingDQM
// 
/**\class EcalAdjustFETimingDQM EcalAdjustFETimingDQM.cc CalibCalorimetry/EcalTiming/plugins/EcalAdjustFETimingDQM.h

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Seth Cooper,27 1-024,+41227672342,
//         Created:  Mon Sep 26 17:38:06 CEST 2011
// $Id: EcalAdjustFETimingDQM.h,v 1.4 2011/11/14 22:22:23 franzoni Exp $
//
//


#include <memory>
#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TProfile2D.h"

//
// class declaration
//

class EcalAdjustFETimingDQM : public edm::EDAnalyzer {
   public:
      explicit EcalAdjustFETimingDQM(const edm::ParameterSet&);
      ~EcalAdjustFETimingDQM();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      virtual void beginRun(edm::Run const&, edm::EventSetup const&);
      virtual void endRun(edm::Run const&, edm::EventSetup const&);
      virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

      void moveBinsTProfile2D(TProfile2D* myprof);
      void scaleBinsTProfile2D(TProfile2D* myprof, double weight);
      std::string intToString(int num);
      int getRunNumber(std::string fileName);

      // ----------member data ---------------------------
      std::string ebDQMFileName_;
      std::string eeDQMFileName_;
      std::string xmlFileNameBeg_;
      std::string txtFileName_;
      std::string rootFileNameBeg_;
      bool readDelaysFromDB_;
      double minTimeChangeToApply_;
      bool operateInDumpMode_;
};
