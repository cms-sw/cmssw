// +-[ Requirements ]---------------------------------------------------------+
// |  - Labels/InputTags
// |      L    oTTRHBuilder
// |      IT   oDigi
// |      IT   oCluster
// |      IT   oClusterInfo
// |      IT   oTrack
// |
// |  - Event Collections
// |      Digi
// |      Clusters
// |      ClusterInfos
// |      Tracks
// |
// |  - EventSetup Records
// |      Transient Builder
// +------------------------------------------------
//
// Author : Samvel Khalatian (samvel at fnal dot gov)
// Created: 02/09/07
// Licence: GPL

#ifndef ANALYSISEXAMPLES_SISTRIPDETECTORPERFORMANCE_SLICETESTNTUPLEMAKER
#define ANALYSISEXAMPLES_SISTRIPDETECTORPERFORMANCE_SLICETESTNTUPLEMAKER

#include <memory>
#include <string>
#include <vector>

#include "AnalysisExamples/SiStripDetectorPerformance/interface/GetSiStripClusterXTalk.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/GeometryVector/interface/Point3DBase.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Handle.h"

// Save Compile time by forwarding declarations
#include "FWCore/Framework/interface/Frameworkfwd.h"

class TFile;
class TTree;

class DetId;
class SiStripCluster;
class SiStripClusterInfo;
class TrackingRecHit;

namespace edm {
  class InputTag;
}

namespace reco {
  class Track;
}

class SliceTestNtupleMaker: public edm::EDAnalyzer {
  public:
    // Constructor
    SliceTestNtupleMaker( const edm::ParameterSet &roCONFIG);
    virtual ~SliceTestNtupleMaker() {}

  protected:
    // Leave possibility of inheritance
    virtual void beginJob( const edm::EventSetup &roEVENT_SETUP);
    virtual void analyze ( const edm::Event      &roEVENT,
                           const edm::EventSetup &roEVENT_SETUP);
    virtual void endJob  ();

  private:
    typedef std::vector<reco::Track>              VRecoTracks;
    typedef edm::DetSetVector<SiStripCluster>     DSVSiStripClusters;
    typedef edm::DetSetVector<SiStripClusterInfo> DSVSiStripClusterInfos;
    typedef std::map<const SiStripCluster *, 
                     const SiStripClusterInfo *>  MClusterVsClusterInfo;

    // Prevent objects copying
    SliceTestNtupleMaker( const SliceTestNtupleMaker &);
    SliceTestNtupleMaker &operator =( const SliceTestNtupleMaker &);

    void buildClusterVsClusterInfoMap( 
      const DSVSiStripClusters     *poSI_STRIP_CLUSTERS,
      const DSVSiStripClusterInfos *poSI_STRIP_CLUSTER_INFO);

    std::string           oOFileName_;
    std::auto_ptr<TFile>  poOFile_;
    TTree                 *poClusterTree_;
    TTree                 *poTrackTree_;
    MClusterVsClusterInfo oMClusterVsClusterInfo_;

    struct {
      edm::InputTag oITDigi;
      edm::InputTag oITCluster;
      edm::InputTag oITClusterInfo;
      edm::InputTag oITTrack;

      std::string   oTTRHBuilder;
    } oConfig_;

    // --[ Group All Coordinates ]---------------------------------------------
    struct Coords {
      Coords():
        dX    ( -10000),
        dY    ( -10000),
        dZ    ( -10000),
        dR    ( -10000),
        dPhi  ( -10000),
        dMag  ( -10000),
        dTheta( -10000) {}
      
      template<class T>
        void setPoint( const Point3DBase<float, T> &roPOINT);

      // Local Coordinates
      float dX;      // Cartesian
      float dY;
      float dZ;
      float dR;      // Cylindrical
      float dPhi;
      float dMag;    // Spherical
      float dTheta;
    };

    // --[ Group General Variables ]-------------------------------------------
    struct GenVal {
      GenVal():
        nRun     ( -10),
        nLclEvent( -10),
        nLTime   ( -10)  {}

      int nRun;
      int nLclEvent;
      int nLTime;
    } oGenVal_;

    // --[ Group Detector Variables ]------------------------------------------
    struct DetVal {
      DetVal():
        // 0 - stands for non-Physical SubDetector
        eSubDet( StripSubdetector::SubDetector( 0)), 
        nLayer       ( -10),
        nWheel       ( -10),
        nRing        ( -10),
        nSide        ( -10),
        nPetal       ( -10),
        nPetalFwBw   ( -10),
        nRod         ( -10),
        nRodFwBw     ( -10),
        nString      ( -10),
        nStringFwBw  ( -10),
        nStringIntExt( -10),
        bStereo      ( false), // Mono by default
        nModule      ( -10),
        nModuleFwBw  ( -10) {}

      void setDet( const DetId &roDET_ID);

      StripSubdetector::SubDetector eSubDet;
      int                           nLayer;
      int                           nWheel;
      int                           nRing;
      int                           nSide;
      int                           nPetal;
      int                           nPetalFwBw;
      int                           nRod;
      int                           nRodFwBw;
      int                           nString;
      int                           nStringFwBw;
      int                           nStringIntExt;
      bool                          bStereo;
      int                           nModule;
      int                           nModuleFwBw;
    };

    // --[ Group Track Variables ]i--------------------------------------------
    struct TrackVal {
      TrackVal():
        bTrack   ( false),
        nTrackNum( -10),
        dMomentum( -10),
        dPt      ( -10),
        dCharge  ( -10),
        dEta     ( -10),
        dPhi     ( -10),
        dHits    ( -10),
        dChi2    ( -10),
        dChi2Norm( -10),
        dNdof    ( -10) {}

      void setTrack( const reco::Track &roTRACK,
                     const int         &rnTRACK_NUM);

      bool  bTrack;
      int   nTrackNum;
      float dMomentum;
      float dPt;
      float dCharge;
      float dEta;
      float dPhi;
      float dHits;
      float dChi2;
      float dChi2Norm;
      float dNdof;
    } oTrackVal_;

    // --[ Group Hit Variables ]-----------------------------------------------
    struct HitVal {
      HitVal(): bMatched( false) {}

      void setHit( const TrackingRecHit  *poHIT,
                   const TrackerGeometry *poTRACKER_GEOMETRY);

      Coords oCoordsLocal;
      Coords oCoordsGlobal;
      bool   bMatched;
      DetVal oDetVal;
    } oHitVal_;

    // --[ Group Cluster Variables ]-------------------------------------------
    struct ClusterVal {
      ClusterVal():
        nFirstStrip( -10),
        dCharge    ( -10),
        dChargeL   ( -10),
        dChargeR   ( -10),
        dNoise     ( -10),
        dBaryCenter( -10),
        dMaxCharge ( -10),
        nWidth     ( -10),
        dXTalk     ( -10) {}

      void setCluster( const SiStripCluster                      *poCLUSTER,
                       const MClusterVsClusterInfo               
                         &roDSVCLUSTER_INFOS,
                       const edm::Handle<extra::DSVSiStripDigis> &roDSVDigis); 

      int    nFirstStrip;
      float  dCharge;
      float  dChargeL;
      float  dChargeR;
      float  dNoise;
      float  dBaryCenter;
      float  dMaxCharge;
      int    nWidth;
      float  dXTalk;
      DetVal oDetVal;
    } oClusterVal_;

}; // End SliceTestNtupleMaker class

template<class T>
  void SliceTestNtupleMaker::Coords::setPoint( 
    const Point3DBase<float, T> &roPOINT) {

  dX     = roPOINT.x();
  dY     = roPOINT.y();
  dZ     = roPOINT.z();
  dR     = sqrt( dX * dX + dY * dY);
  dPhi   = roPOINT.phi();
  dMag   = roPOINT.mag();
  dTheta = roPOINT.theta();
}

#endif // ANALYSISEXAMPLES_SISTRIPDETECTORPERFORMANCE_SLICETESTNTUPLEMAKER
