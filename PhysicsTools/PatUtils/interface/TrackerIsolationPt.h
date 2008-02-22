//
// $Id: TrackerIsolationPt.h,v 1.1 2008/01/15 13:20:58 lowette Exp $
//

#ifndef PhysicsTools_PatUtils_TrackerIsolationPt_h
#define PhysicsTools_PatUtils_TrackerIsolationPt_h

/**
  \class    TrackerIsolationPt TrackerIsolationPt.h "PhysicsTools/PatUtils/interface/TrackerIsolationPt.h"
  \brief    Calculates a lepton's tracker isolation pt

   TrackerIsolationPt calculates a tracker isolation pt in a cone
   around the lepton's direction, without doing track extrapolation

  \author   Steven Lowette
  \version  $Id: TrackerIsolationPt.h,v 1.1 2008/01/15 13:20:58 lowette Exp $
*/

#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"

#include "DataFormats/TrackReco/interface/Track.h"


namespace pat {


  class TrackerIsolationPt {

    public:

      TrackerIsolationPt();
      virtual ~TrackerIsolationPt();

      float calculate(const Electron & theElectron, const edm::View<reco::Track> & theTracks, float isoConeElectron = 0.3) const;
      float calculate(const Muon & theMuon, const edm::View<reco::Track> & theTracks, float isoConeMuon = 0.3) const;

    private:

      float calculate(const reco::Track & theTrack, const edm::View<reco::Track> & theTracks, float isoCone) const;

  };


}

#endif
