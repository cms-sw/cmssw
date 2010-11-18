#ifndef CalibMuon_DTCalibration_DTVDriftT0FitCalibration_h
#define CalibMuon_DTCalibration_DTVDriftT0FitCalibration_h

/** \class DTVDriftT0FitCalibration
 *  Produces histograms from v-drift computation in
 *  segment fit to be used for v-drift calibration
 *
 *  $Date: 2010/11/16 19:06:59 $
 *  $Revision: 1.4 $
 *  \author A. Vilela Pereira
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CalibMuon/DTCalibration/interface/DTSegmentSelector.h"

#include <map>

class DTChamberId;
class TFile;
class TH1F;

class DTVDriftT0FitCalibration : public edm::EDAnalyzer {
public:
  // Constructor
  DTVDriftT0FitCalibration(const edm::ParameterSet& pset);
  // Destructor
  virtual ~DTVDriftT0FitCalibration();

  void beginRun(const edm::Run& run, const edm::EventSetup& setup);
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup);
  void endJob();
  
private:
  typedef std::map<DTChamberId, std::vector<TH1F*> > ChamberHistosMap;
  void bookHistos(DTChamberId);

  DTSegmentSelector select_;

  edm::InputTag  theRecHits4DLabel_;
  bool writeVDriftDB_;
  std::string theCalibChamber_;

  TFile* rootFile_;
  ChamberHistosMap theVDriftHistoMap_;
};
#endif

