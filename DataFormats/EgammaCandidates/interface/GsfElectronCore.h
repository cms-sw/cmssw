#ifndef GsfElectronCore_h
#define GsfElectronCore_h

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include <vector>

namespace reco
 {


/****************************************************************************
 * \class reco::GsfElectronCore
 *
 * Core description of an electron, including a a GsfTrack seeded from an
 * ElectronSeed. The seed was either calo driven, or tracker driven
 * (particle flow). In the latter case, the GsfElectronCore also
 * contains a reference to the pflow supercluster.
 *
 * \author Claude Charlot - Laboratoire Leprince-Ringuet - École polytechnique, CNRS/IN2P3
 * \author David Chamont  - Laboratoire Leprince-Ringuet - École polytechnique, CNRS/IN2P3
 * \author Ursula Berthon - Laboratoire Leprince-Ringuet - École polytechnique, CNRS/IN2P3
 *
 * \version $Id: GsfElectronCore.h,v 1.3 2009/04/06 11:18:05 chamont Exp $
 *
 ****************************************************************************/

//*****************************************************************************
//
// $Log: GsfElectronCore.h,v $
// Revision 1.3  2009/04/06 11:18:05  chamont
// few changes, should not affect users
//
// Revision 1.2  2009/03/24 17:26:27  charlot
// updated provenance and added comments in headers
//
// Revision 1.1  2009/03/20 22:59:16  chamont
// new class GsfElectronCore and new interface for GsfElectron
//
// Revision 1.20  2009/02/14 11:00:26  charlot
// new interface for fiducial regions
//
//*****************************************************************************


class GsfElectronCore {

  public :

    // construction
    GsfElectronCore() ;
    GsfElectronCore( const GsfTrackRef & ) ;
    ~GsfElectronCore() {}

    // accessors
    const GsfTrackRef & gsfTrack() const { return gsfTrack_ ; }
    const SuperClusterRef & superCluster() const
     { return (superCluster_.isNull()?pflowSuperCluster_:superCluster_) ; }
    bool isEcalDriven() const { return ecalDrivenSeed() ; }
    bool isTrackerDriven() const { return trackerDrivenSeed() ; }
    bool ecalDrivenSeed() const { return ecalDrivenSeed_ ; }
    bool trackerDrivenSeed() const { return trackerDrivenSeed_ ; }

    // setters, still useful to GsfElectronSelector.h ??
    void setGsfTrack( const GsfTrackRef & gsfTrack ) { gsfTrack_ = gsfTrack ; }
    void setSuperCluster( const SuperClusterRef & scl ) { superCluster_ = scl ; }

    // pflow eventual additionnal info
    const SuperClusterRef & pflowSuperCluster() const { return pflowSuperCluster_ ; }
    void setPflowSuperCluster( const SuperClusterRef & scl ) { pflowSuperCluster_ = scl ; }

  private :

    GsfTrackRef gsfTrack_ ;
    SuperClusterRef superCluster_ ;
    SuperClusterRef pflowSuperCluster_ ;
    bool ecalDrivenSeed_ ;
    bool trackerDrivenSeed_ ;

 } ;

 }

#endif
