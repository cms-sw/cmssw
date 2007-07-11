#ifndef CalibMuon_DTT0Calibration_H
#define CalibMuon_DTT0Calibration_H

/** \class DTT0Calibration
 *  Analyzer class computes the mean and RMS of t0 from pulses.
 *  Those values are written in the DB with cell granularity. The
 *  mean value for each channel is normalized to the mean within the chamber.
 *  The TProfile histograms filled with cell granularity are also saved to root file.
 *
 *  $Date: 2006/06/08 15:41:58 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - INFN Torino
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"

#include <string>
#include <map>


namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class TFile;
class TProfile;
class DTT0;

class DTT0Calibration : public edm::EDAnalyzer {
public:
  /// Constructor
  DTT0Calibration(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTT0Calibration();

  // Operations

  /// Fill the histos with t0 times (by channel)
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  /// Get the mean and rhe RMS of the t0 from the histo and write them to the DB with channel granularity
  void endJob();


protected:

private:
  // Generate the histo name
  std::string getHistoName(const DTLayerId& lId) const;

  // Print computed t0s
  void dumpT0Map(const DTT0* t0s) const;

  // Debug flag
  bool debug;

  // The label used to retrieve digis from the event
  std::string digiLabel;

  // The file which will contain the time boxes
  TFile *theFile;
  
  // Map of the histograms by Layer
  std::map<DTLayerId, TProfile*> theHistoMap;

};
#endif

