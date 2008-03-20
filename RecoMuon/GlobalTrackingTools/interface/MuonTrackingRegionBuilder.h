#ifndef RecoMuon_TrackingTools_MuonTrackingRegionBuilder_H
#define RecoMuon_TrackingTools_MuonTrackingRegionBuilder_H

/** \class MuonTrackingRegionBuilder
 *
 *  Build a TrackingRegion around a standalone muon
 *
 *
 *  $Date: 2008/03/05 21:12:54 $
 *  $Revision: 1.6 $
 *
 *  \author A. Everett - Purdue University
 *  \author A. Grelli -  Purdue University, Pavia University 
 */

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

class MuonServiceProxy;
class RectangularEtaPhiTrackingRegion;

namespace edm {class ParameterSet; class Event;}

//              ---------------------
//              -- Class Interface --
//              ---------------------

class MuonTrackingRegionBuilder {
  
  public:
 
    /// constructor
    MuonTrackingRegionBuilder(const edm::ParameterSet&, 
                              const MuonServiceProxy*);
  
    /// destructor
    virtual ~MuonTrackingRegionBuilder() {}
  
    /// define tracking region
    RectangularEtaPhiTrackingRegion* region(const reco::TrackRef&) const;

    /// define tracking region
    RectangularEtaPhiTrackingRegion* region(const reco::Track&) const;

    /// pass the Event to the algo at each event
    virtual void setEvent(const edm::Event&);

  private:

    edm::InputTag theBeamSpotTag;   // beam spot
    edm::InputTag theVertexCollTag; // vertex collection

    const edm::Event* theEvent;
    const MuonServiceProxy* theService;

    bool useFixedRegion;
    bool useVertex;

    double theTkEscapePt;
    double theNsigmaEta,theNsigmaDz,theNsigmaPhi;
  
    double theEtaRegionPar1; 
    double theEtaRegionPar2;
    double thePhiRegionPar1;
    double thePhiRegionPar2;

    double thePhiMin;
    double theEtaMin;
    double theDeltaR,theHalfZ;
    double thePhiFixed,theEtaFixed;

    GlobalPoint theVertexPos;

};
#endif
