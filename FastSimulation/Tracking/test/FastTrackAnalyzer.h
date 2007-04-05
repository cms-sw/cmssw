//Alexander.Schmidt@cern.ch
//March 2007

#ifndef FastTrackAnalyzer_h
#define FastTrackAnalyzer_h

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

//--- for SimHit
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

//simtrack
//#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2DCollection.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include <iostream>
#include <string>
#include <map>
#include <set>
#include <vector>
#include "TString.h"

class TH1F;

class FastTrackAnalyzer : public edm::EDAnalyzer {

  
 public:
  explicit FastTrackAnalyzer(const edm::ParameterSet& conf);
  
  virtual ~FastTrackAnalyzer();
  
  virtual void analyze(const edm::Event& event, const edm::EventSetup& setup);
  virtual   void beginJob( const edm::EventSetup& );
  virtual  void endJob();
 private:
  void makeHitsPlots(TString prefix, const SiTrackerGSRecHit2D * rechit, PSimHit * simHit,  const TrackingRecHit *);


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
    double TIB_Pos_AxisLim ;
    double TID_Res_AxisLim ;
    double TID_Pos_AxisLim ;
    double TOB_Res_AxisLim ;
    double TOB_Pos_AxisLim ;
    double TEC_Res_AxisLim ;
    double TEC_Pos_AxisLim ;
    int NumTracks_AxisLim;

    double TIB_Err_AxisLim;
    double TID_Err_AxisLim;
    double TOB_Err_AxisLim;
    double TEC_Err_AxisLim;

    int iEventCounter;
};

#endif
