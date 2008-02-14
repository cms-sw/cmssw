#ifndef RecoMuon_TrackingTools_MuonTrackingRegionBuilder_H
#define RecoMuon_TrackingTools_MuonTrackingRegionBuilder_H

/** \class MuonTrackingRegionBuilder
 *  Base class for the Muon reco TrackingRegion Builder
 *
 *  $Date: 2007/08/15 15:15:28 $
 *  $Revision: 1.1 $
 *  \author A. Everett - Purdue University
 *  \author A. Grelli -  Purdue University, Pavia University 
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"

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

 private:
  bool theFixedFlag;
  
  const MuonServiceProxy * theService;
  
  double HalfZRegion_size; 
  double Delta_R_Region;
  double TkEscapePt;
  double Nsigma_eta;
  double Nsigma_phi;
  
  double Eta_Region_parameter1; 
  double Eta_Region_parameter2;
  double Phi_Region_parameter1;
  double Phi_Region_parameter2;

  double Phi_minimum;
  double Eta_minimum;
  double Phi_fixed;
  double Eta_fixed;

  GlobalPoint theVertexPos;
  GlobalError theVertexErr;

};
#endif
