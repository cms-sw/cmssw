#ifndef MuonReco_Muon_h
#define MuonReco_Muon_h
/** \class reco::Muon
 *  
 * A reconstructed Muon ment to be stored in the AOD
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Muon.h,v 1.15 2006/04/20 14:29:18 llista Exp $
 *
 */
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

namespace reco {
 
  class Muon {
  public:
    /// spatial vector
    typedef math::XYZVector Vector;
    /// point in the space
    typedef math::XYZPoint Point;
    /// index to default track
    enum defaultMomentumEstimate {
      i_tracker = 0, i_standAlone, i_combined
    };
    /// default constructor
    Muon() { }
    /// constructor from values
    Muon( const TrackRef & t, const TrackRef & s, const TrackRef & c, defaultMomentumEstimate d );
    /// default momentum estimate
    defaultMomentumEstimate defaultFitEstimate() const {
      if ( default_ == i_tracker ) return i_tracker;
      if ( default_ == i_combined ) return i_combined;
      if ( default_ == i_standAlone ) return i_standAlone;
      throw edm::Exception( edm::errors::InvalidReference, "Invalid default muon reference" );
    }
    /// reference to Track reconstructed in the tracker only
    const TrackRef & track() const { return track_; }
    /// reference to Track reconstructed in the muon detector only
    const TrackRef & standAlone() const { return standAlone_; }
    /// reference to Track reconstructed in both tracked and muon detector
    const TrackRef & combined() const { return combined_; }
    /// first iterator to RecHits
    const TrackRef & defaultFit() const { 
      if ( default_ == i_tracker ) return track();
      if ( default_ == i_combined ) return combined();
      if ( default_ == i_standAlone ) return standAlone();
      throw edm::Exception( edm::errors::InvalidReference, "Invalid default muon reference" );
    }
    /// fit chi-squared
    double chi2() const { return defaultFit()->chi2(); }
    /// number of degrees of freedom of the fit 
    double ndof() const { return defaultFit()->ndof(); }
    /// chi-squared divided by n.d.o.f. 
    double normalizedChi2 () const { return defaultFit()->normalizedChi2(); }
    /// number of hits found
    unsigned short found() const { return defaultFit()->found(); }
    ///  number of invalid hits
    unsigned short invalid() const { return defaultFit()->invalid(); }
    /// number of hits lost 
    unsigned short lost() const { return defaultFit()->lost(); }
    /// electric charge
    int charge() const { return defaultFit()->charge(); }
    /// momentum vector
    Vector momentum() const { return defaultFit()->momentum(); }
    /// position of point of closest approach to the beamline
    Point vertex() const { return defaultFit()->vertex(); }
    /// default transverse momentum
    double pt() const { return defaultFit()->pt(); }
    /// momentum vector magnitude
    double p() const { return momentum().R(); }
    /// x coordinate of momentum vector
    double px() const { return momentum().X(); }
    /// y coordinate of momentum vector
    double py() const { return momentum().Y(); }
    /// z coordinate of momentum vector
    double pz() const { return momentum().Z(); }
    /// azimuthal angle of momentum vector
    double phi() const { return momentum().Phi(); }
    /// pseudorapidity of momentum vector
    double eta() const { return momentum().Eta(); }
    /// polar angle of momentum vector
    double theta() const { return momentum().Theta(); }
    /// x coordinate of point of closest approach to the beamline
    double x() const { return vertex().X(); }
    /// y coordinate of point of closest approach to the beamline
    double y() const { return vertex().Y(); }
    /// z coordinate of point of closest approach to the beamline
    double z() const { return vertex().Z(); }

    /// set reference to associated Ecal SuperCluster
    void setSuperCluster( const SuperClusterRef & ref ) { superCluster_ = ref; }
    /// reference to associated Ecal SuperCluster
    const SuperClusterRef & superCluster() const { return superCluster_; }

  private:
    /// reference to Track reconstructed in the tracker only
    TrackRef track_;
    /// reference to Track reconstructed in the muon detector only
    TrackRef standAlone_;
    /// reference to Track reconstructed in both tracked and muon detector
    TrackRef combined_;
    /// reference to associated Ecal SuperCluster
    SuperClusterRef superCluster_;
    /// default track
    unsigned char default_;
};

}

#endif
