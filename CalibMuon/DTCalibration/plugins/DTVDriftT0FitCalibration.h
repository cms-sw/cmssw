#ifndef CalibMuon_DTCalibration_DTVDriftT0FitCalibration_h
#define CalibMuon_DTCalibration_DTVDriftT0FitCalibration_h

/** \class DTVDriftT0FitCalibration
 *  Produces histograms from v-drift computation in
 *  segment fit to be used for v-drift calibration
 *
 *  $Date: 2010/11/18 11:40:19 $
 *  $Revision: 1.1 $
 *  \author A. Vilela Pereira
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CalibMuon/DTCalibration/interface/DTSegmentSelector.h"

#include <map>

class DTChamberId;
class TFile;
class TH1F;
class TH2F;

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
  typedef std::map<DTChamberId, std::vector<TH1F*> > ChamberHistosMapTH1F;
  typedef std::map<DTChamberId, std::vector<TH2F*> > ChamberHistosMapTH2F;
  void bookHistos(DTChamberId);

  DTSegmentSelector select_;

  edm::InputTag  theRecHits4DLabel_;
  bool writeVDriftDB_;
  std::string theCalibChamber_;

  TFile* rootFile_;
  ChamberHistosMapTH1F theVDriftHistoMapTH1F_;
  ChamberHistosMapTH2F theVDriftHistoMapTH2F_;
};
#endif

