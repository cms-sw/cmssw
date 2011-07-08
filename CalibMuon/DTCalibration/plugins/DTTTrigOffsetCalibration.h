#ifndef DTTTrigOffsetCalibration_H
#define DTTTrigOffsetCalibration_H

/** \class DTTTrigOffsetCalibration
 *  No description available.
 *
 *  $Date: 2010/02/16 10:03:23 $
 *  $Revision: 1.2 $
 *  \author A. Vilela Pereira
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

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
  void bookHistos(DTChamberId);
  
protected:

private:

  typedef std::map<DTChamberId, std::vector<TH1F*> > ChamberHistosMap;

  // The label used to retrieve 4D segments from the event
  edm::InputTag  theRecHits4DLabel_;

  // The file which will contain the t0-seg histograms
  TFile *theFile_;

  // Do t0-seg correction to ttrig
  bool doTTrigCorrection_;

  // TTrig map 
  const DTTtrig *tTrigMap;  

  // Check for noisy channels
  bool checkNoisyChannels_;

  // Map of superlayers and t0-seg histos
  ChamberHistosMap theT0SegHistoMap_;

  // Maximum value for the 4D Segment chi2
  double theMaxChi2_;

  // Maximum incident angle for Phi Seg 
  double theMaxPhiAngle_;

  // Maximum incident angle for Theta Seg
  double theMaxZAngle_;

  // Choose the chamber you want to calibrate
  std::string theCalibChamber_;

  std::string dbLabel;

};
#endif

