#ifndef BTauReco_CombinedBTagTrack_h
#define BTauReco_CombinedBTagTrack_h

#include "DataFormats/TrackReco/interface/Track.h"
#include "Geometry/CommonDetAlgo/interface/Measurement1D.h"

namespace reco {
  class CombinedBTagTrack {
    public:
      /**
       * A data class used for storing b-tagging info
       * that can be associated with a track
       */
      CombinedBTagTrack();

      CombinedBTagTrack( const reco::TrackRef & ref, bool usedInSVX, double rapidity, 
                 double d0Sign, double jetDistance, const Measurement1D & ip2d,
                 const Measurement1D & ip3d, bool aboveCharmMass );

      CombinedBTagTrack( const reco::TrackRef & ref,
                 double d0Sign, double jetDistance,
                 const Measurement1D & ip2d, const Measurement1D & ip3d );

      void print() const;
      double chi2() const;
      double pt() const;
      double eta() const;
      double d0() const;
      double d0Error() const;
      int nHitsTotal() const;
      int nHitsPixel() const;
      bool firstHitPixel() const;
      const reco::TrackRef & trackRef() const;
      bool usedInSVX() const;
      double rapidity() const;
      double d0Sign() const;
      double jetDistance() const;
      Measurement1D ip2D() const;
      Measurement1D ip3D() const;
      bool aboveCharmMass() const;
      bool isValid() const;

      void setUsedInSVX ( bool );
      void setAboveCharmMass ( bool );
      void setRapidity ( double );

    private:
      reco::TrackRef trackRef_;
      bool   usedInSVX_;    // part of a secondary vertex?
      double rapidity_;
      double d0Sign_;       // same, but lifetime signed
      double jetDistance_;
      Measurement1D ip2D_; // lifetime-signed 2D impact parameter
      Measurement1D ip3D_; // lifetime-signed 3D impact parameter
      bool aboveCharmMass_;  /**
         * tracks are sorted by lifetime-signed 2D impact
         * parameter significance. Starting from the
         * highest significance, the invariant mass
         * of the tracks is calculated (using Pion mass
         * hypothesis). If the mass exceeds a threshold,
         * this flag is set to true.
         */
      bool isValid_;
  };
}

#endif
