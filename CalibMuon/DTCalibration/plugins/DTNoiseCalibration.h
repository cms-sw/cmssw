#ifndef DTNoiseCalibration_H
#define DTNoiseCalibration_H

/*
 * \file DTNoiseCalibration.h
 *
 * $Date: 2010/07/19 22:17:25 $
 * $Revision: 1.9 $
 * \author G. Mila - INFN Torino
 *
*/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <string>
#include <vector>
#include <map>

class DTGeometry;
class DTSuperLayerId;
class DTLayerId; 
class DTWireId;
class DTTtrig;
class TFile;
class TH2F;
class TH1F;

class DTNoiseCalibration: public edm::EDAnalyzer{

 public:
  /// Constructor
  DTNoiseCalibration(const edm::ParameterSet& ps);
  /// Destructor
  virtual ~DTNoiseCalibration();

  void beginJob();
  void beginRun(const edm::Run& run, const edm::EventSetup& setup );
  void analyze(const edm::Event& e, const edm::EventSetup& c);
  void endJob();

private:
  std::string getChannelName(const DTWireId& wId) const;
  // Get the name of the layer
  std::string getLayerName(const DTLayerId& lId) const;
  // Get the name of the superLayer
  std::string getSuperLayerName(const DTSuperLayerId& dtSLId) const;

  edm::InputTag digiLabel_;
  bool useTimeWindow_;
  int triggerWidth_;
  int timeWindowOffset_;
  double maximumNoiseRate_;
  bool useAbsoluteRate_; 

  /*bool fastAnalysis;
  int wh;
  int sect;*/

  bool readDB_;
  int defaultTtrig_;
  std::string dbLabel_;

  std::vector<DTWireId> wireIdWithHisto_;
  unsigned int lumiMax_;

  int nevents_;
  //int counter;

  // Get the DT Geometry
  edm::ESHandle<DTGeometry> dtGeom_;
  // tTrig map
  edm::ESHandle<DTTtrig> tTrigMap_;

  TFile* rootFile_;
  // TDC digi distribution
  TH1F* hTDCTriggerWidth_;
  // Map of the occupancy histograms by layer
  std::map<DTLayerId, TH1F*> theHistoOccupancyMap_;
  // Map of occupancy by lumi by wire
  std::map<DTWireId, TH1F*> theHistoOccupancyVsLumiMap_; 
  // Map of the histograms with the number of events per evt per wire
  //std::map<DTLayerId, TH2F*> theHistoEvtPerWireMap_;
  // Map of skipped histograms
  //std::map<DTLayerId, int> skippedPlot;
};
#endif
