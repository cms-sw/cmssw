#ifndef CalibMuon_DTTTrigCalibration_H
#define CalibMuon_DTTTrigCalibration_H

/** \class DTTTrigCalibration
 *  Analyzer class which fills time box plots with SL granularity
 *  for t_trig computation, fits the rising edge and write results to DB.
 *  The time boxes are written to file.
 *
 *  \author G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"

#include <string>
#include <map>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}  // namespace edm

class TFile;
class TH1F;
class DTTimeBoxFitter;
class DTTTrigBaseSync;
class DTTtrig;
class DTStatusFlag;
class DTStatusFlagRcd;

class DTTTrigCalibration : public edm::one::EDAnalyzer<> {
public:
  /// Constructor
  DTTTrigCalibration(const edm::ParameterSet& pset);

  /// Destructor
  ~DTTTrigCalibration() override;

  // Operations

  /// Fill the time boxes
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override;

  /// Fit the time box rising edge and write the resulting ttrig to the DB
  void endJob() override;

protected:
private:
  // Generate the time box name
  std::string getTBoxName(const DTSuperLayerId& slId) const;
  // Generate the time box name
  std::string getOccupancyName(const DTLayerId& slId) const;

  // Print computed ttrig
  void dumpTTrigMap(const DTTtrig* tTrig) const;

  // Plot computed ttrig
  void plotTTrig(const DTTtrig* tTrig) const;

  // Debug flag
  bool debug;

  // The token used to retrieve digis from the event
  edm::EDGetTokenT<DTDigiCollection> digiToken;

  // The TDC time-window
  int maxTDCCounts;
  //The maximum number of digis per layer
  int maxDigiPerLayer;

  // The file which will contain the time boxes
  TFile* theFile;

  // Map of the histograms by SL
  std::map<DTSuperLayerId, TH1F*> theHistoMap;
  std::map<DTLayerId, TH1F*> theOccupancyMap;

  // Switch for t0 subtraction
  bool doSubtractT0;
  // Switch for checking of noisy channels
  bool checkNoisyChannels;
  //card to switch on/off the DB writing
  bool findTMeanAndSigma;
  // the kfactor to be uploaded in the ttrig DB
  double kFactor;

  // The fitter
  std::unique_ptr<DTTimeBoxFitter> theFitter;
  // The module for t0 subtraction
  std::unique_ptr<DTTTrigBaseSync> theSync;  //FIXME: should be const
  edm::ESGetToken<DTStatusFlag, DTStatusFlagRcd> theStatusMapToken;
};
#endif
