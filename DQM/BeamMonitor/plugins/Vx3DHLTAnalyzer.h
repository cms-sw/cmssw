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
// $Id: Vx3DHLTAnalyzer.h,v 1.14 2010/10/14 23:05:23 wmtan Exp $
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

#include <TF3.h>

#include <iostream>
#include <fstream>
#include <vector>


// #################
// # Fit variables #
// #################
#define DIM 3
void Gauss3DFunc(int& /*npar*/, double* /*gin*/, double& fval, double* par, int /*iflag*/);
typedef struct
{
  double x;
  double y;
  double z;
  double Covariance[DIM][DIM];
} VertexType;
std::vector<VertexType> Vertices;
bool considerVxCovariance;
unsigned int counterVx; // Counts the number of vertices taken into account for the fit
double maxTransRadius;  // Max transverse radius in which the vertices must be [cm]
double maxLongLength;   // Max longitudinal length in which the vertices must be [cm]
double xPos,yPos,zPos;  // x,y,z approximate positions of the beam spot
double pi;
// ######################
// # cfg file parameter #
// ######################
double VxErrCorr;       // Coefficient to compensate the under-estimation of the vertex errors


class Vx3DHLTAnalyzer : public edm::EDAnalyzer {
   public:
      explicit Vx3DHLTAnalyzer(const edm::ParameterSet&);
      ~Vx3DHLTAnalyzer();


   private:
      virtual void beginJob();
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual unsigned int HitCounter(const edm::Event& iEvent);
      virtual char* formatTime(const time_t& t);
      virtual int MyFit(std::vector<double>* vals);
      virtual void reset(std::string ResetType);
      virtual void writeToFile(std::vector<double>* vals,
			       edm::TimeValue_t BeginTimeOfFit,
			       edm::TimeValue_t EndTimeOfFit,
			       unsigned int BeginLumiOfFit,
			       unsigned int EndLumiOfFit,
			       int dataType);
      virtual void beginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, 
					const edm::EventSetup& iSetup);
      virtual void endLuminosityBlock(const edm::LuminosityBlock& lumiBlock,
				      const edm::EventSetup& iSetup);
      virtual void endJob();


      // #######################
      // # cfg file parameters #
      // #######################
      edm::InputTag vertexCollection;
      bool debugMode;
      unsigned int nLumiReset;
      bool dataFromFit;
      unsigned int minNentries;
      double xRange;
      double xStep;
      double yRange;
      double yStep;
      double zRange;
      double zStep;
      std::string fileName;


      // ##############
      // # Histograms #
      // ##############
      MonitorElement* mXlumi;
      MonitorElement* mYlumi;
      MonitorElement* mZlumi;
      
      MonitorElement* sXlumi;
      MonitorElement* sYlumi;
      MonitorElement* sZlumi;
      
      MonitorElement* dxdzlumi;
      MonitorElement* dydzlumi;

      MonitorElement* Vx_X;
      MonitorElement* Vx_Y;
      MonitorElement* Vx_Z;
      
      MonitorElement* Vx_ZX;
      MonitorElement* Vx_ZY;
      MonitorElement* Vx_XY;
      
      MonitorElement* goodVxCounter;
      MonitorElement* goodVxCountHistory;

      MonitorElement* hitCounter;
      MonitorElement* hitCountHistory;

      MonitorElement* reportSummary;
      MonitorElement* reportSummaryMap;
      
      MonitorElement* fitResults;

      // ######################
      // # Internal variables #
      // ######################
      ofstream outputFile;
      ofstream outputDebugFile;
      edm::TimeValue_t beginTimeOfFit;
      edm::TimeValue_t endTimeOfFit;
      unsigned int nBinsHistoricalPlot;
      unsigned int nBinsWholeHistory;
      unsigned int runNumber;
      unsigned int lumiCounter;
      unsigned int lumiCounterHisto;
      unsigned int totalHits;
      unsigned int maxLumiIntegration;
      unsigned int prescaleHistory;
      unsigned int numberGoodFits;
      unsigned int numberFits;
      unsigned int beginLumiOfFit;
      unsigned int endLumiOfFit;
      unsigned int lastLumiOfFit;
      double minVxDoF;
      bool internalDebug;
};

#endif
