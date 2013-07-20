#ifndef DTVDriftCalibration_H
#define DTVDriftCalibration_H

/** \class DTVDriftCalibration
 *  No description available.
 *
 *  $Date: 2013/05/23 15:28:45 $
 *  $Revision: 1.6 $
 *  \author M. Giunta
 */


#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"

#include "CalibMuon/DTCalibration/interface/vDriftHistos.h"
#include "CalibMuon/DTCalibration/interface/DTTMax.h"
#include "CalibMuon/DTCalibration/interface/DTSegmentSelector.h"

#include "DTCalibrationMap.h"

#include <string>
#include <vector>

namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}

class TFile;
class DTMeanTimerFitter;

class DTVDriftCalibration : public edm::EDAnalyzer {
public:
  /// Constructor
  DTVDriftCalibration(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTVDriftCalibration();

  // Operations

  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  void endJob();
  
protected:

private:

  DTSegmentSelector select_;

  // The class containing TMax information
  typedef DTTMax::TMax TMax;
 
  // class to create/manage histos for each partition (SL) 
  class cellInfo{
  public:
    cellInfo(const TString& name) {
      histos = new hTMaxCell(name);
    }  
   
    ~cellInfo() {
      delete histos;
    }

    void add(const std::vector<const TMax*>& tMaxes);
    void update() {addedCells.clear();}
    hTMaxCell* getHists() {return histos;}
    
  private: 
    cellInfo(){};
    cellInfo(const cellInfo&){};
    
    std::vector<dttmaxenums::TMaxCells> addedCells;
    hTMaxCell* histos;
  };

  h2DSegm *h2DSegmRZ;
  h2DSegm *h2DSegmRPhi;
  h4DSegm *h4DSegmAllCh;

  // Divide cellInfo by given granularity (to be implemented)
  // DTVDriftCalibration::cellInfo* partition(const DTWireId& wireId); 

  // Specify the granularity for the TMax histograms
  enum TMaxGranularity {byChamber, bySL, byPartition};
  TMaxGranularity theGranularity;
 
  // The label used to retrieve 4D segments from the event
  edm::InputTag theRecHits4DLabel;

  // Debug flag
  bool debug;
  
  // The label used to retrieve digis from the event
  std::string digiLabel;
  
  // The file which will contain the tMax histograms
  TFile *theFile;

  // The fitter
  DTMeanTimerFitter *theFitter;

  // Perform the vDrift and t0 evaluation or just fill the
  //  tMaxHists (if you read the dataset in different jobs)
  bool findVDriftAndT0;

  // The name of the output text file
  std::string theVDriftOutputFile;

  // Map of wires and cellInfo with coarse granularity
  std::map<DTWireId, cellInfo*> theWireIdAndCellMap;

  // Switch for checking of noisy channels
  //bool checkNoisyChannels;

  // The module for t0 subtraction
  DTTTrigBaseSync *theSync;//FIXME: should be const

  // parameter set for DTCalibrationMap constructor
  edm::ParameterSet theCalibFilePar;

  // Maximum value for the 4D Segment chi2
  //double theMaxChi2;

  // Maximum incident angle for Phi Seg 
  //double theMaxPhiAngle;

  // Maximum incident angle for Theta Seg
  //double theMaxZAngle;

  // Choose the chamber you want to calibrate
  std::string theCalibChamber;

};
#endif

