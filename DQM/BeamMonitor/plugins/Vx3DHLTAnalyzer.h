#ifndef Vx3DHLTAnalyzer_H
#define Vx3DHLTAnalyzer_H

// -*- C++ -*-
//
// Package:    Vx3DHLTAnalyzer
// Class:      Vx3DHLTAnalyzer
// 
/**\class Vx3DHLTAnalyzer Vx3DHLTAnalyzer.cc interface/Vx3DHLTAnalyzer.h

 Description:     beam-spot monitor entirely based on pixel detector information
 Implementation:  the monitoring is based on a 3D fit to the vertex cloud
*/
//
// Original Author:  Mauro Dinardo, 28 S-012, +41-22-767-8302,
//         Created:  Tue Feb 23 13:15:31 CET 2010

// -*- C++ -*-
//
// Package:    Vx3DHLTAnalyzer
// Class:      Vx3DHLTAnalyzer
// 


#include <memory>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include <TF3.h>

#include <iostream>
#include <fstream>
#include <vector>


using namespace std;
using namespace reco;
using namespace edm;


// #################
// # Fit variables #
// #################
#define DIM 3
double Gauss3DFunc(const double* par);
typedef struct
{
  double x;
  double y;
  double z;
  double Covariance[DIM][DIM];
} VertexType;
vector<VertexType> Vertices;
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


class Vx3DHLTAnalyzer : public EDAnalyzer {
   public:
      explicit Vx3DHLTAnalyzer (const ParameterSet&);
      ~Vx3DHLTAnalyzer();


   private:
      virtual void beginJob ();
      virtual void analyze (const Event&, const EventSetup&);
      virtual unsigned int HitCounter (const Event& iEvent);
      virtual string formatTime (const time_t& t);
      virtual int MyFit (vector<double>* vals);
      virtual void reset (string ResetType);
      virtual void writeToFile (vector<double>* vals,
				TimeValue_t BeginTimeOfFit,
				TimeValue_t EndTimeOfFit,
				unsigned int BeginLumiOfFit,
				unsigned int EndLumiOfFit,
				int dataType);
      virtual void beginLuminosityBlock (const LuminosityBlock& lumiBlock, const EventSetup& iSetup);
      virtual void endLuminosityBlock (const LuminosityBlock& lumiBlock, const EventSetup& iSetup);
      virtual void endJob ();
      virtual void beginRun (const Run& iRun, const EventSetup& iSetup);


      // #######################
      // # cfg file parameters #
      // #######################
      EDGetTokenT<VertexCollection> vertexCollection;
      EDGetTokenT<SiPixelRecHitCollection> pixelHitCollection;
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
      string fileName;


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
      TimeValue_t beginTimeOfFit;
      TimeValue_t endTimeOfFit;
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
