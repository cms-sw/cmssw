//--------------------------------------------------------------------------------------------------
// $Id: ConversionTrack.h,v 1.6 2010/11/22 01:54:46 bendavid Exp $
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

class Trajectory;
namespace reco
{
  class ConversionTrack
  {
    public:
      ConversionTrack() : isTrackerOnly_(false), isArbitratedEcalSeeded_(false), isArbitratedMerged_(false),
                          isArbitratedMergedEcalGeneral_(false) {}
      ConversionTrack(const TrackBaseRef &trk) : 
        track_(trk), isTrackerOnly_(false), isArbitratedEcalSeeded_(false), isArbitratedMerged_(false),
        isArbitratedMergedEcalGeneral_(false) {}
      virtual ~ConversionTrack() {}
    
      const reco::Track      *track()                           const { return track_.get();         }
      const TrackBaseRef     &trackRef()                        const { return track_;               }
      const edm::Ref<std::vector<Trajectory> > &trajRef()       const { return traj_;                }
      void                    setTrajRef(edm::Ref<std::vector<Trajectory> > tr) { traj_ = tr;        }
      void                    setIsTrackerOnly(bool b)                { isTrackerOnly_ = b;          }
      void                    setIsArbitratedEcalSeeded(bool b)       { isArbitratedEcalSeeded_ = b; }      
      void                    setIsArbitratedMerged(bool b)           { isArbitratedMerged_ = b;     }
      void                    setIsArbitratedMergedEcalGeneral(bool b) { isArbitratedMergedEcalGeneral_ = b; }      
      bool                    isTrackerOnly() const                   { return isTrackerOnly_;}
      bool                    isArbitratedEcalSeeded() const          { return isArbitratedEcalSeeded_;}
      bool                    isArbitratedMerged() const              { return isArbitratedMerged_;}
      bool                    isArbitratedMergedEcalGeneral() const   { return isArbitratedMergedEcalGeneral_;}




    private:
      TrackBaseRef        track_; //ptr to track
      edm::Ref<std::vector<Trajectory> > traj_;  //reference to a trajectory
      bool                isTrackerOnly_; //from general tracks collection
      bool                isArbitratedEcalSeeded_; //from in out or out-in ecal-seeded collections (arbitrated)
      bool                isArbitratedMerged_; //is arbitrated among all input collections
      bool                isArbitratedMergedEcalGeneral_; //is arbitrated among ecal-seeded and generalTracks     
  };
}
#endif
