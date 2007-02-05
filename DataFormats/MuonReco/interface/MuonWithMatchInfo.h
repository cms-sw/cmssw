#ifndef MuonReco_MuonWithMatchInfo_h
#define MuonReco_MuonWithMatchInfo_h

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonWithMatchInfoFwd.h"
#include "DataFormats/DetId/interface/DetId.h"

/** \class reco::MuonWithMatchInfo
 *  
 * A Muon with particle identification information:
 *  - energy deposition in ECAL, HCAL, HO
 *  - muon segment matching
 *
 * \author Dmytro Kovalskyi, UCSB
 *
 * \version $Id: MuonWithMatchInfo.h,v 1.3 2007/01/30 18:15:20 dmytro Exp $
 *
 */

namespace reco {
   class MuonWithMatchInfo : public Muon {
    public:
      MuonWithMatchInfo() { }
      /// muon constructor
      MuonWithMatchInfo(  Charge, const LorentzVector &, const Point & = Point( 0, 0, 0 ) );
      /// energy deposition
      struct MuonEnergy {
	 float had;   // energy deposited in HCAL
	 float em;    // energy deposited in ECAL
	 float ho;    // energy deposited in HO
      };
      /// muon segment matching information in local coordinates
      struct MuonSegmentMatch {
	 float x;         // X position of the matched segment
	 float y;         // Y position of the matched segment
	 float xErr;      // uncertainty in X
	 float yErr;      // uncertainty in Y
	 float dXdZ;      // dX/dZ of the matched segment
	 float dYdZ;      // dY/dZ of the matched segment
	 float dXdZErr;   // uncertainty in dX/dZ
	 float dYdZErr;   // uncertainty in dY/dZ
      };
      /// muon chamber matching information
      struct MuonChamberMatch {
	 std::vector<MuonSegmentMatch> segmentMatches;  // segments matching propagated track trajectory
	 std::vector<MuonSegmentMatch> truthMatches;  // SimHit projection matching propagated track trajectory
	 float edgeX;      // distance to closest edge in X (negative - inside, positive - outside)
	 float edgeY;      // distance to closest edge in Y (negative - inside, positive - outside)
	 float x;          // X position of the track
	 float y;          // Y position of the track
	 float xErr;       // propagation uncertainty in X
	 float yErr;       // propagation uncertainty in Y
	 float dXdZ;       // dX/dZ of the track
	 float dYdZ;       // dY/dZ of the track
	 float dXdZErr;    // propagation uncertainty in dX/dZ
	 float dYdZErr;    // propagation uncertainty in dY/dZ
	 DetId id;          // chamber ID
      };
      /// get energy deposition information
      MuonEnergy calEnergy() const { return calEnergy_; }
      /// set energy deposition information
      void setCalEnergy( const MuonEnergy& calEnergy ) { calEnergy_ = calEnergy; }
      /// get muon matching information
      std::vector<MuonChamberMatch>& matches() { return muMatches_;}
      const std::vector<MuonChamberMatch>& matches() const { return muMatches_;	}
      /// set muon matching information
      void setMatches( const std::vector<MuonChamberMatch>& matches ) { muMatches_ = matches; }
      /// number of chambers
      int numberOfChambers() const { return muMatches_.size(); }
      /// get number of chambers with matched segments
      int numberOfMatches() const;
      /// X-residual of the first matched segment i-th matched chamber
      /// Note: It is expected that either segment matches are sorted by some
      /// best match criteria or an arbitration is performed to select only match.
      /// If no segment is found dX = 999999.
      float dX(uint i) const;
      /// Y-residual of the first matched segment i-th matched chamber 
      float dY(uint i) const;
      /// propagation uncertainty on X for i-th matched chamber
      float dXErr(uint i) const;
      /// propagation uncertainty on Y for i-th matched chamber
      float dYErr(uint i) const;
      // const CaloTowerRefs& traversedTowers() {return traversedTowers_;}
      
    private:
      /// energy deposition 
      MuonEnergy calEnergy_;
      /// Information on matching between tracks and segments
      std::vector<MuonChamberMatch> muMatches_;
      // vector of references to traversed towers. Could be useful for
      // correcting the missing transverse energy.  This assumes that
      // the CaloTowers will be kept in the AOD.  If not, we need something else.
      // CaloTowerRefs traversedTowers_;
      // Still missing trigger information
   };
}

#endif
