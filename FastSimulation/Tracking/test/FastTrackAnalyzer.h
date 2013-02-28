//Alexander.Schmidt@cern.ch
//March 2007

#ifndef FastTrackAnalyzer_h
#define FastTrackAnalyzer_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

//--- for SimHit
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"

#include <iostream>
#include <string>
#include <map>
#include <set>
#include <vector>
#include "TString.h"

class TH1F;
class TrackerTopology;

class FastTrackAnalyzer : public edm::EDAnalyzer {

  
 public:
  explicit FastTrackAnalyzer(const edm::ParameterSet& conf);
  
  virtual ~FastTrackAnalyzer();
  
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup) override;
  virtual void beginRun(edm::Run const& , edm::EventSetup const& ) override;
  virtual void endJob() override;
 private:
  void makeHitsPlots(TString prefix, const SiTrackerGSRecHit2D * rechit, const PSimHit * simHit, 
		     int numpartners, const TrackerTopology *tTopo);
  
  std::pair<LocalPoint,LocalVector> projectHit( const PSimHit& hit, const StripGeomDetUnit* stripDet,
						const BoundPlane& plane, int thesign) ;
  
  edm::ParameterSet conf_;
  const TrackerGeometry * trackerG;
  edm::ESHandle<MagneticField>          theMagField;
  
  // for the SimHits
  std::vector<std::string> trackerContainers;
  
  std::map<TString, TH1F*> hMap;
  
  
  double PXB_Res_AxisLim ;   
  double PXF_Res_AxisLim ;   
  double PXB_RecPos_AxisLim ;
  double PXF_RecPos_AxisLim ;
  double PXB_SimPos_AxisLim ;
  double PXF_SimPos_AxisLim ;
  double PXB_Err_AxisLim;
  double PXF_Err_AxisLim;
  
  double TIB_Res_AxisLim ;
  double TIB_Resy_AxisLim ;
  double TIB_Pos_AxisLim ;
  double TID_Res_AxisLim ;
  double TID_Resy_AxisLim ;
  double TID_Pos_AxisLim ;
  double TOB_Res_AxisLim ;
  double TOB_Resy_AxisLim ;
  double TOB_Pos_AxisLim ;
  double TEC_Res_AxisLim ;
  double TEC_Resy_AxisLim ;
  double TEC_Pos_AxisLim ;
  int NumTracks_AxisLim;
  
  double TIB_Err_AxisLim;
  double TID_Err_AxisLim;
  double TOB_Err_AxisLim;
  double TEC_Err_AxisLim;
  double TIB_Erry_AxisLim;
  double TID_Erry_AxisLim;
  double TOB_Erry_AxisLim;
  double TEC_Erry_AxisLim;
  
  int iEventCounter;
  TString outfilename;
  std::string trackProducer;

  edm::InputTag simVertexContainerTag;
  edm::InputTag siTrackerGSRecHit2DCollectionTag;
};

#endif
