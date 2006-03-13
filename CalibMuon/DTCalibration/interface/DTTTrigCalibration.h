#ifndef CalibMuon_DTTTrigCalibration_H
#define CalibMuon_DTTTrigCalibration_H

/** \class DTTTrigCalibration
 *  Analyzer class which fills time box plots with SL granularity
 *  for t_trig computation, fits the rising edge and write results to DB.
 *  The time boxes are written to file.
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"

#include <string>
#include <map>


namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class TFile;
class TH1F;
class DTTimeBoxFitter;

class DTTTrigCalibration : public edm::EDAnalyzer {
public:
  /// Constructor
  DTTTrigCalibration(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTTTrigCalibration();

  // Operations

  /// Fill the time boxes
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  /// Fit the time box rising edge and write the resulting ttrig to the DB
  void endJob();


protected:

private:
  // Generate the time box name
  std::string getTBoxName(const DTSuperLayerId& slId) const;

  // Debug flag
  bool debug;

  // The label used to retrieve digis from the event
  std::string digiLabel;

  // The file which will contain the time boxes
  TFile *theFile;
  
  // Map of the histograms by SL
  std::map<DTSuperLayerId, TH1F*> theHistoMap;

  // The fitter
  DTTimeBoxFitter *theFitter;

  // Parameters for DB
  std::string theConnect;
  std::string theCatalog;
  std::string theTag;
  unsigned int theMessageLevel;
  unsigned int theAuthMethod;
  std::string theCoralUser;
  std::string theCoralPasswd;
  //   int theMaxRun;
  //   int theMinRun;
};
#endif

