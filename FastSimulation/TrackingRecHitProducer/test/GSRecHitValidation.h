//Alexander.Schmidt@cern.ch
//March 2007

#ifndef GSRecHitValidation_h
#define GSRecHitValidation_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

//--- for SimHit
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSMatchedRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"

#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

#include <iostream>
#include <string>
#include <map>
#include <set>
#include <vector>
#include "TString.h"

class TH1F;
class TrackerTopology;

class GSRecHitValidation : public edm::EDAnalyzer {

  
 public:
  explicit GSRecHitValidation(const edm::ParameterSet& conf);
  virtual ~GSRecHitValidation();  
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);
  virtual void beginJob();
  virtual void endJob();

 private:
  void fillHitsPlots(TString prefix, const SiTrackerGSMatchedRecHit2D * rechit, PSimHit * simHit, const TrackerTopology *tTopo);
  edm::ParameterSet conf_;
  const TrackerGeometry * trackerG;
    
  // for the SimHits
  std::vector<std::string> trackerContainers;
  
  std::map<TString, TH1F*> hMap;
  
  //PXB 
  double PXB_Res_AxisLim ;   
  double PXB_RecPos_AxisLim ;
  double PXB_SimPos_AxisLim ;
  double PXB_Err_AxisLim;
  
  //PXF
  double PXF_Res_AxisLim ;   
  double PXF_RecPos_AxisLim ;
  double PXF_SimPos_AxisLim ;
  double PXF_Err_AxisLim;

  //TIB
  double TIB_Pos_x_AxisLim; 
  double TIB_Pos_y_AxisLim; 
  double TIB_Res_x_AxisLim; 
  double TIB_Res_y_AxisLim; 
  double TIB_Pull_x_AxisLim;
  double TIB_Pull_y_AxisLim;
  
  //TOB
  double TOB_Pos_x_AxisLim; 
  double TOB_Pos_y_AxisLim; 
  double TOB_Res_x_AxisLim; 
  double TOB_Res_y_AxisLim; 
  double TOB_Pull_x_AxisLim;
  double TOB_Pull_y_AxisLim;
  
  //TID
  double TID_Pos_x_AxisLim; 
  double TID_Pos_y_AxisLim; 
  double TID_Res_x_AxisLim; 
  double TID_Res_y_AxisLim; 
  double TID_Pull_x_AxisLim;
  double TID_Pull_y_AxisLim;

  //TEC
  double TEC_Pos_x_AxisLim; 
  double TEC_Pos_y_AxisLim; 
  double TEC_Res_x_AxisLim; 
  double TEC_Res_y_AxisLim; 
  double TEC_Pull_x_AxisLim;
  double TEC_Pull_y_AxisLim;

  int iEventCounter;
  TString outfilename;

  edm::InputTag matchedHitCollectionInputTag_;
  edm::InputTag hitCollectionInputTag_;
};

#endif
