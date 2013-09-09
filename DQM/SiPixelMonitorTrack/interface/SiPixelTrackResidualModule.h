// Package:    SiPixelMonitorTrack
// Class:      SiPixelTrackResidualModule
// 
// class SiPixelTrackResidualModule SiPixelTrackResidualModule.h 
//       DQM/SiPixelMonitorTrack/src/SiPixelTrackResidualModule.h
//
// Description: SiPixel hit-to-track residual data quality monitoring modules
// Implementation: prototype -> improved -> never final - end of the 1st step 
//
// Original Author: Shan-Huei Chuang
//         Created: Fri Mar 23 18:41:42 CET 2007
//         Updated by Lukas Wehrli (plots for clusters on/off track added)
// $Id: SiPixelTrackResidualModule.h,v 1.5 2013/02/04 13:45:49 merkelp Exp $


#ifndef SiPixelMonitorTrack_SiPixelTrackResidualModule_h
#define SiPixelMonitorTrack_SiPixelTrackResidualModule_h


#include <boost/cstdint.hpp>

#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementVector.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"


class SiPixelTrackResidualModule { 
  public:
    SiPixelTrackResidualModule();
    SiPixelTrackResidualModule(const uint32_t);
   ~SiPixelTrackResidualModule();

   void book(const edm::ParameterSet&, bool reducedSet=true, int type=0, bool isUpgrade=false);
   void fill(const Measurement2DVector&, bool reducedSet=true, bool modon=true, bool ladon=true, bool layon=true, bool phion = true, bool bladeon=true, bool diskon=true, bool ringon=true);
   void fill(const SiPixelCluster &clust, bool onTrack, double corrCharge, bool reducedSet,  bool modon, bool ladon, bool layon, bool phion, bool bladeon, bool diskon, bool ringon); 
   void nfill(int onTrack, int offTrack, bool reducedSet,  bool modon, bool ladon, bool layon, bool phion, bool bladeon, bool diskon, bool ringon);
  
  private:
    uint32_t id_; 
    bool bBookTracks;


    MonitorElement* meResidualX_;
    MonitorElement* meResidualY_;
    MonitorElement* meNClusters_onTrack_;
    MonitorElement* meCharge_onTrack_;
    MonitorElement* meSize_onTrack_;
    MonitorElement* meSizeX_onTrack_;
    MonitorElement* meSizeY_onTrack_;
    MonitorElement* meNClusters_offTrack_;
    MonitorElement* meCharge_offTrack_;
    MonitorElement* meSize_offTrack_;
    MonitorElement* meSizeX_offTrack_;
    MonitorElement* meSizeY_offTrack_;

    //barrel
    MonitorElement* meResidualXLad_;
    MonitorElement* meResidualYLad_;
    MonitorElement* meNClusters_onTrackLad_;
    MonitorElement* meCharge_onTrackLad_;
    MonitorElement* meSize_onTrackLad_;
    MonitorElement* meSizeX_onTrackLad_;
    MonitorElement* meSizeY_onTrackLad_;
    MonitorElement* meNClusters_offTrackLad_;
    MonitorElement* meCharge_offTrackLad_;
    MonitorElement* meSize_offTrackLad_;
    MonitorElement* meSizeX_offTrackLad_;
    MonitorElement* meSizeY_offTrackLad_;

    MonitorElement* meResidualXLay_;
    MonitorElement* meResidualYLay_;
    MonitorElement* meNClusters_onTrackLay_;
    MonitorElement* meCharge_onTrackLay_;
    MonitorElement* meSize_onTrackLay_;
    MonitorElement* meSizeX_onTrackLay_;
    MonitorElement* meSizeY_onTrackLay_;
    MonitorElement* meNClusters_offTrackLay_;
    MonitorElement* meCharge_offTrackLay_;
    MonitorElement* meSize_offTrackLay_;
    MonitorElement* meSizeX_offTrackLay_;
    MonitorElement* meSizeY_offTrackLay_;

    MonitorElement* meResidualXPhi_;
    MonitorElement* meResidualYPhi_;
    MonitorElement* meNClusters_onTrackPhi_;
    MonitorElement* meCharge_onTrackPhi_;
    MonitorElement* meSize_onTrackPhi_;
    MonitorElement* meSizeX_onTrackPhi_;
    MonitorElement* meSizeY_onTrackPhi_;
    MonitorElement* meNClusters_offTrackPhi_;
    MonitorElement* meCharge_offTrackPhi_;
    MonitorElement* meSize_offTrackPhi_;
    MonitorElement* meSizeX_offTrackPhi_;
    MonitorElement* meSizeY_offTrackPhi_;

    //forward
    MonitorElement* meResidualXBlade_;
    MonitorElement* meResidualYBlade_;
    MonitorElement* meNClusters_onTrackBlade_;
    MonitorElement* meCharge_onTrackBlade_;
    MonitorElement* meSize_onTrackBlade_;
    MonitorElement* meSizeX_onTrackBlade_;
    MonitorElement* meSizeY_onTrackBlade_;
    MonitorElement* meNClusters_offTrackBlade_;
    MonitorElement* meCharge_offTrackBlade_;
    MonitorElement* meSize_offTrackBlade_;
    MonitorElement* meSizeX_offTrackBlade_;
    MonitorElement* meSizeY_offTrackBlade_;

    MonitorElement* meResidualXDisk_;
    MonitorElement* meResidualYDisk_;
    MonitorElement* meNClusters_onTrackDisk_;
    MonitorElement* meCharge_onTrackDisk_;
    MonitorElement* meSize_onTrackDisk_;
    MonitorElement* meSizeX_onTrackDisk_;
    MonitorElement* meSizeY_onTrackDisk_;
    MonitorElement* meNClusters_offTrackDisk_;
    MonitorElement* meCharge_offTrackDisk_;
    MonitorElement* meSize_offTrackDisk_;
    MonitorElement* meSizeX_offTrackDisk_;
    MonitorElement* meSizeY_offTrackDisk_;

    MonitorElement* meResidualXRing_;
    MonitorElement* meResidualYRing_;
    MonitorElement* meNClusters_onTrackRing_;
    MonitorElement* meCharge_onTrackRing_;
    MonitorElement* meSize_onTrackRing_;
    MonitorElement* meSizeX_onTrackRing_;
    MonitorElement* meSizeY_onTrackRing_;
    MonitorElement* meNClusters_offTrackRing_;
    MonitorElement* meCharge_offTrackRing_;
    MonitorElement* meSize_offTrackRing_;
    MonitorElement* meSizeX_offTrackRing_;
    MonitorElement* meSizeY_offTrackRing_;
};

#endif
