#ifndef Vx3DHLTAnalyzer_H
#define Vx3DHLTAnalyzer_H

/*
  \File Vx3DHLTAnalyzer.h
  \Display Beam-spot monitor entirely based on pixel detector information
           the monitoring is based on a 3D fit to the vertex cloud
  \Author Mauro Dinardo
  \Version $ Revision: 3.5 $
  \Date $ Date: 2010/23/02 13:15:00 $
*/

#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"

#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include <TText.h>

#include <iostream>
#include <fstream>
#include <vector>

// #################
// # Fit variables #
// #################
#define DIM 3
typedef struct {
  double x;
  double y;
  double z;
  double Covariance[DIM][DIM];
} VertexType;

class Vx3DHLTAnalyzer : public DQMOneLumiEDAnalyzer<> {
public:
  Vx3DHLTAnalyzer(const edm::ParameterSet&);
  ~Vx3DHLTAnalyzer() override;

protected:
  double Gauss3DFunc(const double* par);

private:
  void analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) override;
  void dqmBeginLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const edm::EventSetup& iSetup) override;
  void dqmEndLuminosityBlock(const edm::LuminosityBlock& lumiBlock, const edm::EventSetup& iSetup) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  unsigned int HitCounter(const edm::Event& iEvent);
  std::string formatTime(const time_t& t);
  int MyFit(std::vector<double>* vals);
  void reset(std::string ResetType);
  void writeToFile(std::vector<double>* vals,
                   edm::TimeValue_t BeginTimeOfFit,
                   edm::TimeValue_t EndTimeOfFit,
                   unsigned int BeginLumiOfFit,
                   unsigned int EndLumiOfFit,
                   int dataType);
  void printFitParams(const std::vector<double>& fitResults);

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
  double VxErrCorr;  // Coefficient to compensate the under-estimation of the vertex errors
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

  MonitorElement* Vx_X_Fit;
  MonitorElement* Vx_Y_Fit;
  MonitorElement* Vx_Z_Fit;

  MonitorElement* Vx_X_Cum;
  MonitorElement* Vx_Y_Cum;
  MonitorElement* Vx_Z_Cum;

  MonitorElement* Vx_ZX_Cum;
  MonitorElement* Vx_ZY_Cum;
  MonitorElement* Vx_XY_Cum;

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

  std::vector<VertexType> Vertices;
  bool considerVxCovariance;
  unsigned int counterVx;   // Counts the number of vertices taken into account for the fit
  double maxTransRadius;    // Max transverse radius in which the vertices must be [cm]
  double maxLongLength;     // Max longitudinal length in which the vertices must be [cm]
  double xPos, yPos, zPos;  // x,y,z approximate positions of the beam spot
  double pi;
};

#endif
