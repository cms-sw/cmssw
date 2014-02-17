#ifndef CalibMuon_DTT0Calibration_H
#define CalibMuon_DTT0Calibration_H

/** \class DTT0Calibration
 *  Analyzer class computes the mean and RMS of t0 from pulses.
 *  Those values are written in the DB with cell granularity. The
 *  mean value for each channel is normalized to a reference time common to all the sector.
 *  The t0 of wires in odd layers are corrected for the relative difference between 
 *  odd and even layers 
 *
 *  $Date: 2012/04/10 17:55:08 $
 *  $Revision: 1.6 $
 *  \author S. Bolognesi - INFN Torino
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <string>
#include <vector>
#include <map>

class TFile;
class TH1I;
class TH1D;
class DTT0;

class DTT0Calibration : public edm::EDAnalyzer {
public:
  /// Constructor
  DTT0Calibration(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTT0Calibration();

  // Operations

  /// Fill the maps with t0 (by channel)
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  /// Compute the mean and the RMS of the t0 from the maps and write them to the DB with channel granularity
  void endJob();


protected:

private:
  // Generate the histo name
  std::string getHistoName(const DTWireId& wId) const;
  std::string getHistoName(const DTLayerId& lId) const;

  // Debug flag
  bool debug;

  // The label used to retrieve digis from the event
  std::string digiLabel;

  // The root file which contain the histos per layer
  TFile *theFile;
  // The root file which will contain the histos per wire (for the given layer)
  TFile *theOutputFile;

  //The event counter
  unsigned int nevents;
  //Number of events to be used for the t0 per layer histos
  unsigned int eventsForLayerT0;
  //Number of events to be used for the t0 reference per wire
  unsigned int eventsForWireT0;

  //Reject digis if far from TP peak
  unsigned int rejectDigiFromPeak;

  //Acceptance of t0 w.r.t. reference peak
  double tpPeakWidth;

  // Write relative t0's with respect to mean t0's in chamber
  bool correctByChamberMean_;

  //The wheels,sector to be calibrated (default All)
  std::string theCalibWheel;
  int selWheel;
  std::string theCalibSector;
  int selSector;

  // Map of the histos and graph by layer
  std::map<DTLayerId, TH1I*> theHistoLayerMap;
  //Histo with t0 mean per layer for all the sector
  TH1D* hT0SectorHisto;
  
  //Layer with histos for each wire
  std::vector<DTWireId> wireIdWithHistos;
  std::vector<std::string> cellsWithHistos;

  //Maps with t0, sigma, number of digi per wire
  std::map<DTWireId,double> theAbsoluteT0PerWire;
  std::map<DTWireId,double> theRelativeT0PerWire;
  std::map<DTWireId,double> theSigmaT0PerWire;
  std::map<DTWireId,int> nDigiPerWire;
  std::map<DTWireId,int> nDigiPerWire_ref;
  std::map<DTWireId,double> mK;
  std::map<DTWireId,double> mK_ref;
  std::map<DTWireId,double> qK;
  //Map with histo per wire for the chosen layer
  std::map<DTWireId,TH1I*> theHistoWireMap;
  std::map<DTWireId,TH1I*> theHistoWireMap_ref;
  //Map with mean and RMS of t0 per layer
  std::map<std::string,double> theT0LayerMap;
  std::map<std::string,double> theSigmaT0LayerMap;

  //DTGeometry used to loop on the SL in the endJob
  edm::ESHandle<DTGeometry> dtGeom;
};
#endif

