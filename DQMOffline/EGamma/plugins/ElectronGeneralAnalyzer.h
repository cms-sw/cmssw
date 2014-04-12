
#ifndef DQMOffline_EGamma_ElectronGeneralAnalyzer_h
#define DQMOffline_EGamma_ElectronGeneralAnalyzer_h

#include "DQMOffline/EGamma/interface/ElectronDqmAnalyzerBase.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

class MagneticField ;

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"


class ElectronGeneralAnalyzer : public ElectronDqmAnalyzerBase
 {
  public:

    explicit ElectronGeneralAnalyzer( const edm::ParameterSet & conf ) ;
    virtual ~ElectronGeneralAnalyzer() ;

    virtual void book() ;
    virtual void analyze( const edm::Event & e, const edm::EventSetup & c ) ;

  private:

    //=========================================
    // parameters
    //=========================================

    // collection input tags
    edm::EDGetTokenT<reco::GsfElectronCollection> electronCollection_;
    edm::EDGetTokenT<reco::SuperClusterCollection> matchingObjectCollection_;
    edm::EDGetTokenT<reco::GsfTrackCollection> gsftrackCollection_;
    edm::EDGetTokenT<reco::TrackCollection> trackCollection_;
    edm::EDGetTokenT<reco::VertexCollection> vertexCollection_;
    edm::EDGetTokenT<reco::BeamSpot> beamSpotTag_;

    // for trigger
    edm::EDGetTokenT<edm::TriggerResults> triggerResults_;
    //std::vector<std::string > HLTPathsByName_;
//
//    //=========================================
//    // general attributes and utility methods
//    //=========================================
//
//    bool trigger( const edm::Event & e ) ;
//    std::vector<unsigned int> HLTPathsByIndex_;

    //=========================================
    // histograms
    //=========================================

    MonitorElement * h2_ele_beamSpotXvsY ;
    MonitorElement * py_ele_nElectronsVsLs ;
    MonitorElement * py_ele_nClustersVsLs ;
    MonitorElement * py_ele_nGsfTracksVsLs ;
    MonitorElement * py_ele_nTracksVsLs ;
    MonitorElement * py_ele_nVerticesVsLs ;
    MonitorElement * h1_ele_triggers ;

 } ;

#endif



