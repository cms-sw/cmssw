#ifndef RecoMuon_TrackingTools_MuonTrackingRegionBuilder_H
#define RecoMuon_TrackingTools_MuonTrackingRegionBuilder_H

/** \class MuonTrackingRegionBuilder
 *  Base class for the Muon reco TrackingRegion Builder
 *
 *  $Date: 2008/03/05 21:12:54 $
 *  $Revision: 1.6 $
 *  \author A. Everett - Purdue University
 *  \author A. Grelli -  Purdue University, Pavia University 
 */

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

namespace edm {class Event;}

class MuonServiceProxy;
class RectangularEtaPhiTrackingRegion;

//namespace reco {class Track; class TrackRef;}

class MuonTrackingRegionBuilder {
  
 public:
  
  /// constructor
  MuonTrackingRegionBuilder(const edm::ParameterSet&, const MuonServiceProxy*);
  
  /// destructor
  virtual ~MuonTrackingRegionBuilder() {}
  
  RectangularEtaPhiTrackingRegion* region(const reco::TrackRef&) const;

  RectangularEtaPhiTrackingRegion* region(const reco::Track&) const;

  /// pass the Event to the algo at each event
  virtual void setEvent(const edm::Event&);


 private:

  edm::InputTag theBeamSpotTag; //beam spot
  edm::InputTag theVertexCollTag;   // Vertexing

  const edm::Event* theEvent;
  const MuonServiceProxy * theService;

  bool    theFixedFlag,EnableBeamSpot,usePixelVertex;
  double  TkEscapePt;
  double  Nsigma_eta,Nsigma_Dz,Nsigma_phi ;
  
  double  Eta_Region_parameter1; 
  double  Eta_Region_parameter2;
  double  Phi_Region_parameter1;
  double  Phi_Region_parameter2;

  double  Phi_minimum,Eta_minimum;
  double  Delta_R_Region,HalfZRegion_size;
  double  Phi_fixed,Eta_fixed;

  GlobalPoint theVertexPos;
  GlobalPoint vertexPosiBS;
};
#endif
