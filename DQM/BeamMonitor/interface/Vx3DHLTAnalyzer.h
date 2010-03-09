#ifndef Vx3DHLTAnalyzer_H
#define Vx3DHLTAnalyzer_H

// -*- C++ -*-
//
// Package:    Vx3DHLTAnalyzer
// Class:      Vx3DHLTAnalyzer
// 
/**\class Vx3DHLTAnalyzer Vx3DHLTAnalyzer.cc interface/Vx3DHLTAnalyzer.h

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Mauro Dinardo,28 S-020,+41227673777,
//         Created:  Tue Feb 23 13:15:31 CET 2010
// $Id: Vx3DHLTAnalyzer.h,v 1.2 2010/03/06 21:38:57 dinardo Exp $
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
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <vector>


using namespace std;


vector<double> xVxValues;
vector<double> yVxValues;
vector<double> zVxValues;


class Vx3DHLTAnalyzer : public edm::EDAnalyzer {
   public:
      explicit Vx3DHLTAnalyzer(const edm::ParameterSet&);
      ~Vx3DHLTAnalyzer();


   private:
      virtual void beginJob();
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob();
      virtual void writeToFile(vector<double>* vals,
			       string fileName,
			       edm::TimeValue_t BeginTimeOfFit,
			       edm::TimeValue_t EndTimeOfFit,
			       unsigned int BeginLumiOfFit,
			       unsigned int EndLumiOfFit);
      virtual void beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, 
					const edm::EventSetup& iSetup);
      virtual void endLuminosityBlock(const edm::LuminosityBlock& lumiBlock,
				      const edm::EventSetup& iSetup);

      // cfg file parameters
      edm::InputTag vertexCollection;
      unsigned int nLumiReset;
      bool dataFromFit;
      int minNentries;
      double xRange;
      double xStep;
      double yRange;
      double yStep;
      double zRange;
      double zStep;
      string fileName;

      // Histograms
      MonitorElement* mXlumi;
      MonitorElement* mYlumi;
      MonitorElement* mZlumi;
      
      MonitorElement* sXlumi;
      MonitorElement* sYlumi;
      MonitorElement* sZlumi;
      
      MonitorElement* dxdzlumi;
      MonitorElement* dydzlumi;

      MonitorElement* RMSsigXlumi;
      MonitorElement* RMSsigYlumi;
      MonitorElement* RMSsigZlumi;

      MonitorElement* Vx_X;
      MonitorElement* Vx_Y;
      MonitorElement* Vx_Z;
      
      MonitorElement* Vx_ZX;
      MonitorElement* Vx_ZY;
      MonitorElement* Vx_XY;
      
      MonitorElement* Vx_ZX_profile;
      MonitorElement* Vx_ZY_profile;

      MonitorElement* reportSummary;
      MonitorElement* reportSummaryMap;
      
      // Internal variables
      unsigned int runNumber;
      unsigned int lumiCounter;
      edm::TimeValue_t beginTimeOfFit;
      edm::TimeValue_t endTimeOfFit;
      unsigned int beginLumiOfFit;
      unsigned int endLumiOfFit;
};

#endif
