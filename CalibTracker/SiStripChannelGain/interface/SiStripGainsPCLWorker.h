// -*- C++ -*-
//
// Package:    CalibTracker/SiStripChannelGain
// Class:      SiStripGainsPCLWorker
// 
/**\class SiStripGainsPCLWorker SiStripGainsPCLWorker.cc 
   Description: Fill DQM histograms with SiStrip Charge normalized to path length
 
*/
//
//  Original Author: L. Quertermont (calibration algorithm)
//  Contributors:    M. Verzetti    (data access)
//                   A. Di Mattia   (PCL multi stream processing and monitoring)
//                   M. Delcourt    (monitoring)
//                   M. Musich      (migration to thread-safe DQMStore access)
//
//  Created:  Wed, 12 Apr 2017 14:46:48 GMT
//

// CMSSW includes
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "CalibFormats/SiStripObjects/interface/SiStripGain.h"
#include "CalibFormats/SiStripObjects/interface/SiStripQuality.h"
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibTracker/Records/interface/SiStripGainRcd.h"
#include "CalibTracker/Records/interface/SiStripQualityRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripApvGain.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/GeometrySurface/interface/RectangularPlaneBounds.h"
#include "DataFormats/GeometrySurface/interface/TrapezoidalPlaneBounds.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h"
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripMatchedRecHit2D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2D.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"

/// user includes
#include "CalibTracker/SiStripChannelGain/interface/APVGainStruct.h"
#include "CalibTracker/SiStripChannelGain/interface/APVGainHelpers.h"

// System includes
#include <unordered_map>

//
// class declaration
//

class SiStripGainsPCLWorker : public  DQMEDAnalyzer {
public:
    explicit SiStripGainsPCLWorker(const edm::ParameterSet&);
     
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
    virtual void beginJob() ;
    void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;   
    void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;
    void analyze(const edm::Event&, const edm::EventSetup&) override;
    virtual void endJob() ;
    
    void processEvent(const TrackerTopology* topo); //what really does the job
    virtual void checkBookAPVColls(const edm::EventSetup& setup);

    std::vector<std::string> dqm_tag_;

    int statCollectionFromMode(const char* tag) const;

    std::vector<MonitorElement*>  Charge_Vs_Index;           /*!< Charge per cm for each detector id */
    std::array< std::vector<APVGain::APVmon>,7 > Charge_1;   /*!< Charge per cm per layer / wheel */
    std::array< std::vector<APVGain::APVmon>,7 > Charge_2;   /*!< Charge per cm per layer / wheel without G2 */
    std::array< std::vector<APVGain::APVmon>,7 > Charge_3;   /*!< Charge per cm per layer / wheel without G1 */
    std::array< std::vector<APVGain::APVmon>,7 > Charge_4;   /*!< Charge per cm per layer / wheel without G1 and G1*/

    std::vector<MonitorElement*>  Charge_Vs_PathlengthTIB;   /*!< Charge vs pathlength in TIB */
    std::vector<MonitorElement*>  Charge_Vs_PathlengthTOB;   /*!< Charge vs pathlength in TOB */
    std::vector<MonitorElement*>  Charge_Vs_PathlengthTIDP;  /*!< Charge vs pathlength in TIDP */
    std::vector<MonitorElement*>  Charge_Vs_PathlengthTIDM;  /*!< Charge vs pathlength in TIDM */
    std::vector<MonitorElement*>  Charge_Vs_PathlengthTECP1; /*!< Charge vs pathlength in TECP thin */
    std::vector<MonitorElement*>  Charge_Vs_PathlengthTECP2; /*!< Charge vs pathlength in TECP thick */
    std::vector<MonitorElement*>  Charge_Vs_PathlengthTECM1; /*!< Charge vs pathlength in TECP thin */
    std::vector<MonitorElement*>  Charge_Vs_PathlengthTECM2; /*!< Charge vs pathlength in TECP thick */

    

    unsigned int NEvent;    
    unsigned int NTrack;
    unsigned int NClusterStrip;
    unsigned int NClusterPixel;
    int NStripAPVs;
    int NPixelDets;
    unsigned int SRun;
    unsigned int ERun;
  
    double       MinTrackMomentum;
    double       MaxTrackMomentum;
    double       MinTrackEta;
    double       MaxTrackEta;
    unsigned int MaxNrStrips;
    unsigned int MinTrackHits;
    double       MaxTrackChiOverNdf;
    int          MaxTrackingIteration;
    bool         AllowSaturation;
    bool         FirstSetOfConstants;
    bool         Validation;
    bool         OldGainRemoving;
    bool         useCalibration;
    bool         doChargeMonitorPerPlane;   /*!< Charge monitor per detector plane */

    std::string  m_DQMdir;                  /*!< DQM folder hosting the charge statistics and the monitor plots */
    std::string  m_calibrationMode;         /*!< Type of statistics for the calibration */
    std::vector<std::string> VChargeHisto;  /*!< Charge monitor plots to be output */

    edm::ESHandle<TrackerGeometry> tkGeom_;
    const TrackerGeometry *bareTkGeomPtr_;   // ugly hack to fill APV colls only once, but checks

    //Data members for processing

    //Event data
    unsigned int                       eventnumber    =0;
    unsigned int                       runnumber      =0;
    const std::vector<bool>*           TrigTech       =nullptr;  edm::EDGetTokenT<std::vector<bool>           > TrigTech_token_;

    // Track data
    const std::vector<double>*         trackchi2ndof  =nullptr;  edm::EDGetTokenT<std::vector<double>         > trackchi2ndof_token_;
    const std::vector<float>*          trackp         =nullptr;  edm::EDGetTokenT<std::vector<float>          > trackp_token_;
    const std::vector<float>*          trackpt        =nullptr;  edm::EDGetTokenT<std::vector<float>          > trackpt_token_;
    const std::vector<double>*         tracketa       =nullptr;  edm::EDGetTokenT<std::vector<double>         > tracketa_token_;
    const std::vector<double>*         trackphi       =nullptr;  edm::EDGetTokenT<std::vector<double>         > trackphi_token_;
    const std::vector<unsigned int>*   trackhitsvalid =nullptr;  edm::EDGetTokenT<std::vector<unsigned int>   > trackhitsvalid_token_;
    const std::vector<int>*            trackalgo      =nullptr;  edm::EDGetTokenT<std::vector<int>   >          trackalgo_token_;

    // CalibTree data
    const std::vector<int>*            trackindex     =nullptr;  edm::EDGetTokenT<std::vector<int>            > trackindex_token_;
    const std::vector<unsigned int>*   rawid          =nullptr;  edm::EDGetTokenT<std::vector<unsigned int>   > rawid_token_;
    const std::vector<double>*         localdirx      =nullptr;  edm::EDGetTokenT<std::vector<double>         > localdirx_token_;
    const std::vector<double>*         localdiry      =nullptr;  edm::EDGetTokenT<std::vector<double>         > localdiry_token_;
    const std::vector<double>*         localdirz      =nullptr;  edm::EDGetTokenT<std::vector<double>         > localdirz_token_;
    const std::vector<unsigned short>* firststrip     =nullptr;  edm::EDGetTokenT<std::vector<unsigned short> > firststrip_token_;
    const std::vector<unsigned short>* nstrips        =nullptr;  edm::EDGetTokenT<std::vector<unsigned short> > nstrips_token_;
    const std::vector<bool>*           saturation     =nullptr;  edm::EDGetTokenT<std::vector<bool>           > saturation_token_;
    const std::vector<bool>*           overlapping    =nullptr;  edm::EDGetTokenT<std::vector<bool>           > overlapping_token_;
    const std::vector<bool>*           farfromedge    =nullptr;  edm::EDGetTokenT<std::vector<bool>           > farfromedge_token_;
    const std::vector<unsigned int>*   charge         =nullptr;  edm::EDGetTokenT<std::vector<unsigned int>   > charge_token_;
    const std::vector<double>*         path           =nullptr;  edm::EDGetTokenT<std::vector<double>         > path_token_;
    const std::vector<double>*         chargeoverpath =nullptr;  edm::EDGetTokenT<std::vector<double>         > chargeoverpath_token_;
    const std::vector<unsigned char>*  amplitude      =nullptr;  edm::EDGetTokenT<std::vector<unsigned char>  > amplitude_token_;
    const std::vector<double>*         gainused       =nullptr;  edm::EDGetTokenT<std::vector<double>         > gainused_token_;
    const std::vector<double>*         gainusedTick   =nullptr;  edm::EDGetTokenT<std::vector<double>         > gainusedTick_token_;

    std::string EventPrefix_; //("");
    std::string EventSuffix_; //("");
    std::string TrackPrefix_; //("track");
    std::string TrackSuffix_; //("");
    std::string CalibPrefix_; //("GainCalibration");
    std::string CalibSuffix_; //("");

    std::vector<std::shared_ptr<stAPVGain> > APVsCollOrdered;
    std::unordered_map<unsigned int, std::shared_ptr<stAPVGain> > APVsColl; 
    
};

inline int
SiStripGainsPCLWorker::statCollectionFromMode(const char* tag) const
{
  std::vector<std::string>::const_iterator it=dqm_tag_.begin();
  while(it!=dqm_tag_.end()) {
    if(*it==std::string(tag)) return it-dqm_tag_.begin();
    it++;
  }
  
  if (std::string(tag)=="") return 0;  // return StdBunch calibration mode for backward compatibility
  
  return None;
}

template<typename T>
inline edm::Handle<T> connect(const T* &ptr, edm::EDGetTokenT<T> token, const edm::Event &evt) {
  edm::Handle<T> handle;
  evt.getByToken(token, handle);
  ptr = handle.product();
  return handle; //return handle to keep alive pointer (safety first)
}
