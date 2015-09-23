#ifndef Vx3DHLTAnalyzer_H
#define Vx3DHLTAnalyzer_H

// -*- C++ -*-
// Package: Vx3DHLTAnalyzer
// Class:   Vx3DHLTAnalyzer

/* 
   Class Vx3DHLTAnalyzer Vx3DHLTAnalyzer.cc plugins/Vx3DHLTAnalyzer.cc

   Description:    beam-spot monitor entirely based on pixel detector information
   Implementation: the monitoring is based on a 3D fit to the vertex cloud
*/

// Original Author: Mauro Dinardo, 28 S-012, +41-22-767-8302
//         Created: Tue Feb 23 13:15:31 CET 2010


#include <memory>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include <TF3.h>

#include <iostream>
#include <fstream>
#include <vector>


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


class Vx3DHLTAnalyzer : public DQMEDAnalyzer
{


 public:
  Vx3DHLTAnalyzer  (const edm::ParameterSet&);
  ~Vx3DHLTAnalyzer ();


 private:
  void analyze              (const edm::Event& iEvent, const edm::EventSetup& iSetup);
  void beginLuminosityBlock (const edm::LuminosityBlock& lumiBlock, const edm::EventSetup& iSetup);
  void endLuminosityBlock   (const edm::LuminosityBlock& lumiBlock, const edm::EventSetup& iSetup);
  void bookHistograms       (DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;

  unsigned int HitCounter (const edm::Event& iEvent);
  std::string formatTime (const time_t& t);
  int MyFit (std::vector<double>* vals);
  void reset (std::string ResetType);
  void writeToFile (std::vector<double>* vals,
		    edm::TimeValue_t BeginTimeOfFit,
		    edm::TimeValue_t EndTimeOfFit,
		    unsigned int BeginLumiOfFit,
		    unsigned int EndLumiOfFit,
		    int dataType);
  void printFitParams (const std::vector<double>& fitResults);


  // #######################
  // # cfg file parameters #
  // #######################
  edm::EDGetTokenT<reco::VertexCollection> vertexCollection;
  edm::EDGetTokenT<SiPixelRecHitCollection> pixelHitCollection;
  bool debugMode;
  bool dataFromFit;
  unsigned int nLumiFit;
  unsigned int maxLumiIntegration;
  unsigned int nLumiXaxisRange;
  unsigned int minNentries;
  double xRange;
  double xStep;
  double yRange;
  double yStep;
  double zRange;
  double zStep;
  double minVxDoF;
  double minVxWgt;
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
  MonitorElement* hitCounter;
  MonitorElement* statusCounter;
  
  MonitorElement* reportSummary;
  MonitorElement* reportSummaryMap;
  
  MonitorElement* fitResults;
  
  
  // ######################
  // # Internal variables #
  // ######################
  std::ofstream outputFile;
  std::ofstream outputDebugFile;
  edm::TimeValue_t beginTimeOfFit;
  edm::TimeValue_t endTimeOfFit;
  unsigned int runNumber;
  unsigned int lumiCounter;
  unsigned int totalHits;
  unsigned int numberGoodFits;
  unsigned int numberFits;
  unsigned int beginLumiOfFit;
  unsigned int endLumiOfFit;
  unsigned int lastLumiOfFit;
  unsigned int nParams;
  bool internalDebug;
};

#endif
