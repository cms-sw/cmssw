//--------------------------------------------------------------------------------------------------
// $Id: StablePart.h,v 1.8 2009/03/20 17:13:33 loizides Exp $
//
// ConversionTrack
//
// Wrapper class holding a pointer to reco::Track plus some various arbitration flags used to
// keep track of overlaps in photon conversion reconstruction.  This class is intended to be used
// to build mixed collections of Track and GsfTracks from different sources and reconstruction
// algorithms to be used for inclusive conversion reconstruction.
//
// Authors: J.Bendavid
//--------------------------------------------------------------------------------------------------

#ifndef EgammaReco_ConversionTrack_h
#define EgammaReco_ConversionTrack_h

#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace reco
{
  class ConversionTrack
  {
    public:
      ConversionTrack() : isTrackerOnly_(false), isArbitratedEcalSeeded_(false), isArbitratedMerged_(false) {}
      ConversionTrack(const TrackBaseRef &trk) : 
        track_(trk), isTrackerOnly_(false), isArbitratedEcalSeeded_(false), isArbitratedMerged_(false) {}
      virtual ~ConversionTrack() {}
    
      const reco::Track      *track()                           const { return track_.get();         }
      const TrackBaseRef     &trackRef()                        const { return track_;               }
      void                    setIsTrackerOnly(bool b)                { isTrackerOnly_ = b;          }
      void                    setIsArbitratedEcalSeeded(bool b)       { isArbitratedEcalSeeded_ = b; }      
      void                    setIsArbitratedMerged(bool b)           { isArbitratedMerged_ = b;     }      

    private:
      TrackBaseRef        track_; //ptr to track
      bool                isTrackerOnly_; //from general tracks collection
      bool                isArbitratedEcalSeeded_; //from in out or out-in ecal-seeded collections (arbitrated)
      bool                isArbitratedMerged_; //is arbitrated among all input collections
  };
}
#endif
