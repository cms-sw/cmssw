#ifndef DTTTrigOffsetCalibration_H
#define DTTTrigOffsetCalibration_H

/** \class DTTTrigOffsetCalibration
 *  No description available.
 *
 *  $Date: 2010/11/16 19:06:59 $
 *  $Revision: 1.4 $
 *  \author A. Vilela Pereira
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CalibMuon/DTCalibration/interface/DTSegmentSelector.h"

#include <map>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class DTChamberId;
class DTTtrig;
class TFile;
class TH1F;

class DTTTrigOffsetCalibration : public edm::EDAnalyzer {
public:
  // Constructor
  DTTTrigOffsetCalibration(const edm::ParameterSet& pset);
  // Destructor
  virtual ~DTTTrigOffsetCalibration();

  void beginRun(const edm::Run& run, const edm::EventSetup& setup);
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup);
  void endJob();
  
private:
  typedef std::map<DTChamberId, std::vector<TH1F*> > ChamberHistosMap;
  void bookHistos(DTChamberId);

  DTSegmentSelector select_;

  edm::InputTag  theRecHits4DLabel_;
  bool doTTrigCorrection_;
  std::string theCalibChamber_;
  std::string dbLabel_;

  TFile* rootFile_;
  const DTTtrig* tTrigMap_;
  ChamberHistosMap theT0SegHistoMap_;
};
#endif

