#ifndef GEMValidation_TFTrack_h
#define GEMValidation_TFTrack_h

#include "GEMCode/GEMValidation/src/BaseMatcher.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/Math/interface/normalizedPhi.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/GeometrySurface/interface/BoundCylinder.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"

#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"

#include <L1Trigger/CSCCommonTrigger/interface/CSCConstants.h>
#include "L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCTFSectorProcessor.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCTrackFinderDataTypes.h"
#include <L1Trigger/CSCTrackFinder/src/CSCTFDTReceiver.h>

#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"

#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"

#include "SimMuon/CSCDigitizer/src/CSCDbStripConditions.h"

#include <Geometry/CSCGeometry/interface/CSCChamberSpecs.h>
#include "Geometry/GEMGeometry/interface/GEMGeometry.h"

#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
// system include files
#include <memory>
#include <cmath>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TH1.h"
#include "TH2.h"
#include "TTree.h"

//#include "TLorentzVector.h"
//#include "DataFormats/Math/interface/LorentzVector.h"
//#include <CLHEP/Vector/LorentzVector.h>

#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
//#include <Geometry/CSCGeometry/interface/CSCLayer.h>
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"


#include "DataFormats/MuonDetId/interface/CSCDetId.h"
//#include <DataFormats/L1CSCTrackFinder/interface/CSCTFConstants.h>
#include <DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h>
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include <DataFormats/CSCDigi/interface/CSCWireDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h>

#include <DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h>
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"

#include "DataFormats/L1GlobalMuonTrigger/interface/L1MuGMTReadoutCollection.h"

#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"

#include <L1Trigger/CSCCommonTrigger/interface/CSCConstants.h>
#include <L1Trigger/CSCTrackFinder/interface/CSCTFPtLUT.h>

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"

#include <vector>
#include <tuple>
#include <iostream>

class TFTrack
{
 public:
  /// constructor
  TFTrack();
  /// copy constructor
  TFTrack(const TFTrack&);
  /// destructor
  ~TFTrack();  

  /// L1 track
  const csc::L1Track* getL1Track() const {return l1track_;}
  /// collection of trigger digis
  std::vector<const CSCCorrelatedLCTDigi* >& 
    getTriggerDigis() const {return return triggerDigis_;} 
  /// collection of MPC LCTs
  std::vector <CSCDetId> getTriggerDigisIds();
  std::vector <std::pair<float, float>> getTriggerEtaPhis();
  std::vector <csctf::TrackStub> getTriggerStubs;
  std::vector <Digi*> mplcts getTriggerMPLCTs();
  std::vector <CSCDetId> getChamberIds; // chamber ids
  
  /// track sign
  bool sign() const {return l1track_->sign();}
  /// bunch crossing 
  int bx() const {return l1track_->bx();}
  /// how many stubs?
  int nStubs(int station);
  /// has stub in muon barrel/endcap
  bool hasStubStation(int wheel);  
  /// has stub in muon barrel?
  bool hasStubBarrel(int wheel);  
  /// has stub in muon endcap?
  bool hasStubEndcap(int station);
  ///   
  //  unsigned dPhi12() const { return 1*(l1trk->ptLUTAddress() & 0xFF);}
  //  unsigned dPhi23() const { return 1*( (l1trk->ptLUTAddress() & 0xF00)>>8 );}
  unsigned ptPacked() const {return pt_packed_;}
  unsigned etaPacked() const {return eta_packed_;}
  unsigned phiPacked() const {return phi_packed_;}
  unsigned qPacked() const {return q_packed_;}
  double pt() const {return pt_;}
  double eta() const {return eta_;}
  double phi() const {return phi_;}
  double dr() const {return dr_;}
  std::vector<bool> deltaOk();
  bool debug() const {return debug_;}

 private:
  const csc::L1Track* l1track_;
    std::vector < const CSCCorrelatedLCTDigi * > trgdigis;
    std::vector < CSCDetId > trgids;
    std::vector < std::pair<float, float> > trgetaphis;
    std::vector < csctf::TrackStub > trgstubs;
    std::vector < Digi* > mplcts;
    std::vector < CSCDetId > ids; // chamber ids
    unsigned phi_packed_;
    unsigned eta_packed_;
    unsigned pt_packed_;
    unsigned q_packed_;
    double phi_;
    double eta_;
    double pt_;
    double dr_;
    std::vector<bool> deltaOk_;
    bool debug_;
};

#endif
