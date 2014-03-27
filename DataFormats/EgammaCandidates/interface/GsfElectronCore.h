#ifndef GsfElectronCore_h
#define GsfElectronCore_h

#include "DataFormats/EgammaCandidates/interface/GsfElectronCoreFwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"

#include <vector>


/****************************************************************************
 * \class reco::GsfElectronCore
 *
 * Core description of an electron, including a a GsfTrack seeded from an
 * ElectronSeed. The seed was either calo driven, or tracker driven
 * (particle flow). In the latter case, the GsfElectronCore also
 * contains a reference to the pflow supercluster.
 *
 ****************************************************************************/


namespace reco
 {

  class GsfElectronCore
   {

    public :

      // construction
      GsfElectronCore() ;
      GsfElectronCore( const GsfTrackRef & ) ;
      GsfElectronCore * clone() const ;
      ~GsfElectronCore() {}

      // accessors
      const GsfTrackRef & gsfTrack() const { return gsfTrack_ ; }
      const SuperClusterRef & superCluster() const
       { return (superCluster_.isNull()?parentSuperCluster_:superCluster_) ; }
      TrackRef ctfTrack() const { return closestCtfTrack_ ; } // get the CTF track best matching the GTF associated to this electron
      float ctfGsfOverlap() const { return ctfGsfOverlap_ ; } // measure the fraction of common hits between the GSF and CTF tracks
      bool ecalDrivenSeed() const { return isEcalDrivenSeed_ ; }
      bool trackerDrivenSeed() const { return isTrackerDrivenSeed_ ; }

      // setters
      void setGsfTrack( const GsfTrackRef & gsfTrack ) { gsfTrack_ = gsfTrack ; }
      void setSuperCluster( const SuperClusterRef & scl ) { superCluster_ = scl ; }
      void setCtfTrack( const TrackRef & closestCtfTrack, float ctfGsfOverlap )
       { closestCtfTrack_ = closestCtfTrack ; ctfGsfOverlap_ = ctfGsfOverlap ; }

      // pflow eventual additionnal info
      const SuperClusterRef & parentSuperCluster() const { return parentSuperCluster_ ; }
      void setParentSuperCluster( const SuperClusterRef & scl ) { parentSuperCluster_ = scl ; }

    private :

      GsfTrackRef gsfTrack_ ;
      SuperClusterRef superCluster_ ;
      SuperClusterRef parentSuperCluster_ ;
      TrackRef closestCtfTrack_ ; // best matching ctf track
      float ctfGsfOverlap_ ; // fraction of common hits between the ctf and gsf tracks
      bool isEcalDrivenSeed_ ;
      bool isTrackerDrivenSeed_ ;

   } ;

 }

//*****************************************************************************
//
// \author David Chamont  - Laboratoire Leprince-Ringuet - École polytechnique, CNRS/IN2P3
// \author Claude Charlot - Laboratoire Leprince-Ringuet - École polytechnique, CNRS/IN2P3
//
//
// Revision 1.11.2.1  2011/03/04 18:22:31  chamont
// backward incompatible cleaning
//
// Revision 1.11  2011/02/18 15:26:05  chamont
// remove the ref to pflow candidate
//
// Revision 1.10  2011/02/16 17:42:53  chamont
// Extend GsfElectron(Core) with new pflow attributes, backward compatible.
//
// Revision 1.9  2011/01/10 17:18:09  chamont
// so to ease the independant production of ecal-driven and tracker-driven gsf electrons
//
// Revision 1.8.6.1  2011/01/10 17:15:35  chamont
// so to ease the independant production of ecal-driven and tracker-driven gsf electrons
//
// Revision 1.8  2010/02/25 15:32:20  chamont
// make GsfElectronFwd.h a real forward header
//
// Revision 1.7  2009/12/11 09:09:41  chamont
// tranform NARROW into OLDNARROW, and add closestCtfTrack to GsfElectronCore
//
// Revision 1.6  2009/10/20 20:57:54  chamont
// restore previous attribute names, so to preserve backward data compatibility
//
// Revision 1.5  2009/10/10 20:07:35  chamont
// remove is*Driven() lethods
//
// Revision 1.4  2009/10/10 09:00:36  chamont
// Add ecalDrivenSeed() to isEcalDriven(), Add trackerDrivenSeed() to isTrackerDriven(), for classes GsfElectron and GsfElectronCore
//
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

#endif
