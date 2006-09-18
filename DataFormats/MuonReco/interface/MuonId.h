#ifndef MuonReco_MuonId_h
#define MuonReco_MuonId_h

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonIdFwd.h"
#include "DataFormats/DetId/interface/DetId.h"

/** \class reco::MuonId MuonId.h DataFormats/MuonReco/interface/MuonId.h
 *  
 * A Muon with particle identification information:
 *  - energy deposition in ECAL, HCAL, HO
 *  - muon segment matching
 *
 * \author Dmytro Kovalskyi, UCSB
 *
 * \version $Id$
 *
 */

namespace reco {
   class MuonId : public Muon {
    public:
      MuonId() { }
      /// muon constructor
      MuonId(  Charge, const LorentzVector &, const Point & = Point( 0, 0, 0 ) );
      /// energy deposition
      struct MuonEnergy {
	 double had;   // energy deposited in HCAL
	 double em;    // energy deposited in ECAL
	 double ho;    // energy deposited in HO
      };
      /// muon segment matching information in local coordinates
      struct MuonMatch {
	 double dX;      // X matching between track and segment
	 double dY;      // Y matching between track and segment
	 double dXErr;   // error in X matching
	 double dYErr;   // error in Y matching
	 double dXdZ;    // dX/dZ matching between track and segment
	 double dYdZ;    // dY/dZ matching between track and segment
	 double dXdZErr; // error in dX/dZ matching
	 double dYdZErr; // error in dY/dZ matching
	 DetId stationId;    // station ID
      };
      /// get energy deposition information
      MuonEnergy calEnergy() const { return calEnergy_; }
      /// set energy deposition information
      void setCalEnergy( const MuonEnergy& calEnergy ) { calEnergy_ = calEnergy; }
      /// get muon segment matching information
      std::vector<MuonMatch> matches() const { return muMatches_;}
      /// set muon segment matching information
      void setMatches( const std::vector<MuonMatch>& matches ) { muMatches_ = matches; }
      /// get number of matched muon segments
      int numberOfMatches() const { return muMatches_.size(); }
      /// X-residual for i-th matched segment
      double dX(int i) const { return muMatches_[i].dX; }
      /// Y-residual for i-th matched segment
      double dY(int i) const { return muMatches_[i].dY; }
      /// error on X for i-th matched segment
      double dXErr(int i) const { return muMatches_[i].dXErr; }
      /// error on Y for i-th matched segment
      double dYErr(int i) const { return muMatches_[i].dYErr; }
      // const CaloTowerRefs& traversedTowers() {return traversedTowers_;}
      
    private:
      /// energy deposition 
      MuonEnergy calEnergy_;
      /// Information on matching between tracks and segments
      std::vector<MuonMatch> muMatches_;
      // Vector of station IDs crossed by the track
      // (This is somewhat redundant with mu_Matches_ but allows
      // to see what segments were "missed".
      // std::vector<DetId> crossedStationID_;
      // vector of references to traversed towers. Could be useful for
      // correcting the missing transverse energy.  This assumes that
      // the CaloTowers will be kept in the AOD.  If not, we need something else.
      // CaloTowerRefs traversedTowers_;
      // Still missing trigger information
   };
}

#endif
