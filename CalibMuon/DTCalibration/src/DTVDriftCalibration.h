#ifndef DTVDriftCalibration_H
#define DTVDriftCalibration_H

/** \class DTVDriftCalibration
 *  No description available.
 *
 *  $Date: 2006/06/22 17:40:39 $
 *  $Revision: 1.2 $
 *  \author M. Giunta
 */


#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "CalibMuon/DTCalibration/src/vDriftHistos.h"
#include "CalibMuon/DTCalibration/src/DTTMax.h"
#include "CalibMuon/DTCalibration/src/DTCalibrationMap.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <string>
#include <vector>


namespace edm {
  class ParameterSet;
  class Event;
  class EventSetup;
}


class DTVDriftCalibration : public edm::EDAnalyzer {
public:
  /// Constructor
  DTVDriftCalibration(const edm::ParameterSet& pset);

  /// Destructor
  virtual ~DTVDriftCalibration();

  // Operations

  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  void endJob();
  
  std::vector<float> evaluateVDriftAndReso (const DTWireId& wireId);
protected:

private:

  // The class containing TMax information
  typedef DTTMax::TMax TMax;
 
  // class to create/manage histos for each partition (SL) 
  class cellInfo{
  public:
    cellInfo(TString name) {
      histos = new hTMaxCell(name);
    }  
   
    ~cellInfo() {
      delete histos;
    }

    void add(std::vector<const TMax*> tMaxes);
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

//   //Divide cellInfo by given granularity (to be implemented)
//   DTVDriftCalibration::cellInfo* partition(const DTWireId& wireId); 

  // Specify the granularity for the TMax histograms
  enum TMaxGranularity {byChamber, bySL, byPartition};
  TMaxGranularity theGranularity;
 
  // The label used to retrieve 4D segments from the event
  std::string  theRecHits4DLabel;

  // Debug flag
  bool debug;
  
  // The label used to retrieve digis from the event
  std::string digiLabel;
  
  // The file which will contain the tMax histograms
  TFile *theFile;

  // Perform the vDrift and t0 evaluation or just fill the
  //  tMaxHists (if you read the dataset in different jobs)
  bool findVDriftAndT0;

  // The name of the output text file
  std::string theVDriftOutputFile;

  // Map of wires and cellInfo with coarse granularity
  std::map<DTWireId, cellInfo*> theWireIdAndCellMap;

  // The module for t0 subtraction
  DTTTrigBaseSync *theSync;//FIXME: should be const

  // parameter set for DTCalibrationMap constructor
  edm::ParameterSet theCalibFilePar;

  // Maximum value for the 4D Segment chi2
  double theMaxChi2;

  // Maximum incident angle for Phi Seg 
  double theMaxPhiAngle;

  // Maximum incident angle for Theta Seg
  double theMaxZAngle;

  // Choose the chamber you want to calibrate
  std::string theCalibChamber;

};
#endif

