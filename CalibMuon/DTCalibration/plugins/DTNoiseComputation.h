#ifndef DTNoiseComputation_H
#define DTNoiseComputation_H

/*
 * \file DTNoiseComputation.h
 *
 * $Date: 2010/01/19 09:51:31 $
 * $Revision: 1.4 $
 * \author G. Mila - INFN Torino
 *
*/

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <FWCore/Framework/interface/EventSetup.h>
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include <FWCore/Framework/interface/ESHandle.h>



#include <string>
#include <map>
#include <vector>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class DTGeometry;
class TFile;
class TH2F;
class TH1F;

class DTNoiseComputation: public edm::EDAnalyzer{
  
 public:
  
  /// Constructor
  DTNoiseComputation(const edm::ParameterSet& ps);
  
  /// Destructor
  virtual ~DTNoiseComputation();

  /// BeginJob
  void beginJob() {}

  void beginRun(const edm::Run&, const edm::EventSetup& setup);

  void analyze(const edm::Event& event, const edm::EventSetup& setup) {}

  /// Endjob
  void endJob();


protected:

private:

  bool debug;
  int counter;
  int MaxEvents;
  bool fastAnalysis;
  
  // Get the DT Geometry
  edm::ESHandle<DTGeometry> dtGeom;

  // The file which contain the occupancy plot and the digi event plot
  TFile *theFile;
  
  // The file which will contain the occupancy plot and the digi event plot
  TFile *theNewFile;

  // Map of label to compute the average noise per layer
  std::map<DTLayerId , bool> toComputeNoiseAverage;

  // Map of the average noise per layer
  std::map<DTWireId , double>  theAverageNoise;

   // Map of the histograms with the number of events per evt per wire
  std::map<DTLayerId, std::vector<TH2F*> > theEvtMap;

  // map of histos with the distance of event per wire
  std::map<DTWireId, TH1F*> theHistoEvtDistancePerWire;
  
  // Map of label for analysis histos
  std::map<DTWireId , bool> toDel;

  // Map of the Time Constants per wire
  std::map<DTWireId , double> theTimeConstant;

  /// Get the name of the layer
  std::string getLayerName(const DTLayerId& lId) const;

  /// Get the name of the superLayer
  std::string getSuperLayerName(const DTSuperLayerId& slId) const;

  /// Get the name of the chamber
  std::string getChamberName(const DTLayerId& lId) const;
  
  // map of histos with the average noise per chamber
  std::map<DTChamberId, TH1F*> AvNoisePerChamber;

  // map of histos with the average integrated noise per chamber
  std::map<DTChamberId, TH1F*> AvNoiseIntegratedPerChamber;

  // map of histos with the average noise per SuperLayer
  std::map<DTSuperLayerId, TH1F*> AvNoisePerSuperLayer;

  // map of histos with the average integrated noise per SuperLayer
  std::map<DTSuperLayerId, TH1F*> AvNoiseIntegratedPerSuperLayer;

  // get the maximum bin number
  int getMaxNumBins(const DTChamberId& chId) const;
  
  // get the Y axis maximum
  double getYMaximum(const DTSuperLayerId& slId) const;

  // map of noisy cell occupancy
  std::map< std::pair<int,int> , TH1F*> noisyC;

  // map of somehow noisy cell occupancy
  std::map< std::pair<int,int> , TH1F*> someHowNoisyC;

};
#endif
