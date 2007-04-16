#ifndef MuonReco_Muon_h
#define MuonReco_Muon_h
/** \class reco::Muon Muon.h DataFormats/MuonReco/interface/Muon.h
 *  
 * A reconstructed Muon.
 * contains reference to three fits:
 *  - tracker alone
 *  - muon detector alone
 *  - combined muon plus tracker
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Muon.h,v 1.27 2007/03/20 12:18:23 llista Exp $
 *
 */
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/MuonReco/interface/MuonChamberMatch.h"

namespace reco {
 
  class Muon : public RecoCandidate {
  public:
    Muon() { }
    /// constructor from values
    Muon(  Charge, const LorentzVector &, const Point & = Point( 0, 0, 0 ) );
    /// create a clone
    Muon * clone() const;
    /// define arbitration schemes
    /// DefaultArbitration is DxArbitration
    enum ArbitrationType { NoArbitration, DefaultArbitration, DxArbitration, DrArbitration };
    
    /// reference to Track reconstructed in the tracker only
    TrackRef track() const { return track_; }
    /// reference to Track reconstructed in the muon detector only
    TrackRef standAloneMuon() const { return standAloneMuon_; }
    /// reference to Track reconstructed in both tracked and muon detector
    TrackRef combinedMuon() const { return combinedMuon_; }
    /// set reference to Track
    void setTrack( const TrackRef & t ) { track_ = t; }
    /// set reference to Track
    void setStandAlone( const TrackRef & t ) { standAloneMuon_ = t; }
    /// set reference to Track
    void setCombined( const TrackRef & t ) { combinedMuon_ = t; }

    /// energy deposition
    struct MuonEnergy {
       float had;    // energy deposited in HCAL
       float em;     // energy deposited in ECAL
       float ho;     // energy deposited in HO
    };
    bool isEnergyValid() const { return energyValid_; }
    /// get energy deposition information
    MuonEnergy getCalEnergy() const { return calEnergy_; }
    /// set energy deposition information
    void setCalEnergy( const MuonEnergy& calEnergy ) { calEnergy_ = calEnergy; energyValid_ = true; }
     
    bool isMatchesValid() const { return matchesValid_; }
    /// get muon matching information
    std::vector<MuonChamberMatch>& getMatches() { return muMatches_;}
    const std::vector<MuonChamberMatch>& getMatches() const { return muMatches_;	}
    /// set muon matching information
    void setMatches( const std::vector<MuonChamberMatch>& matches ) { muMatches_ = matches; matchesValid_ = true; }

    /// number of chambers
    int numberOfChambers() const { return muMatches_.size(); }
    /// get number of chambers with matched segments
    int numberOfMatches( ArbitrationType type = DefaultArbitration ) const;
    /// get bit map of stations with matched segments
    /// bits 0-1-2-3 = DT stations 1-2-3-4
    /// bits 4-5-6-7 = CSC stations 1-2-3-4
    unsigned int stationMask( ArbitrationType type = DefaultArbitration ) const;
    unsigned int stationGapMask( ArbitrationType type = DefaultArbitration ) const;
     
     
  private:
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
    /// reference to Track reconstructed in the tracker only
    TrackRef track_;
    /// reference to Track reconstructed in the muon detector only
    TrackRef standAloneMuon_;
    /// reference to Track reconstructed in both tracked and muon detector
    TrackRef combinedMuon_;
    /// energy deposition 
    MuonEnergy calEnergy_;
    /// Information on matching between tracks and segments
    std::vector<MuonChamberMatch> muMatches_;
    /// vector of references to traversed towers. Could be useful for
    /// correcting the missing transverse energy.  
    CaloTowerRefs traversedTowers_;
    bool energyValid_;
    bool matchesValid_;
    // FixMe: Still missing trigger information

    /// get vector of muon chambers for given station and detector
    const std::vector<const MuonChamberMatch*> getChambers( int station, int muonSubdetId ) const;
    /// get pointers to best segment and corresponding chamber in vector of chambers
    std::pair<const MuonChamberMatch*,const MuonSegmentMatch*> getPair( const std::vector<const MuonChamberMatch*>,
									ArbitrationType type = DefaultArbitration ) const;
     
   public:
     /// get number of segments
     int numberOfSegments( int station, int muonSubdetId, ArbitrationType type = DefaultArbitration ) const;
     /// get deltas between (best) segment and track
     /// If no chamber or no segment returns 999999
     float dX       ( int station, int muonSubdetId, ArbitrationType type = DefaultArbitration ) const;
     float dY       ( int station, int muonSubdetId, ArbitrationType type = DefaultArbitration ) const;
     float dDxDz    ( int station, int muonSubdetId, ArbitrationType type = DefaultArbitration ) const;
     float dDyDz    ( int station, int muonSubdetId, ArbitrationType type = DefaultArbitration ) const;
     float pullX    ( int station, int muonSubdetId, ArbitrationType type = DefaultArbitration, bool includeSegmentError = false ) const;
     float pullY    ( int station, int muonSubdetId, ArbitrationType type = DefaultArbitration, bool includeSegmentError = false ) const;
     float pullDxDz ( int station, int muonSubdetId, ArbitrationType type = DefaultArbitration, bool includeSegmentError = false ) const;
     float pullDyDz ( int station, int muonSubdetId, ArbitrationType type = DefaultArbitration, bool includeSegmentError = false ) const;
     /// get (best) segment information
     /// If no segment returns 999999
     float segmentX       ( int station, int muonSubdetId, ArbitrationType type = DefaultArbitration ) const;
     float segmentY       ( int station, int muonSubdetId, ArbitrationType type = DefaultArbitration ) const;
     float segmentDxDz    ( int station, int muonSubdetId, ArbitrationType type = DefaultArbitration ) const;
     float segmentDyDz    ( int station, int muonSubdetId, ArbitrationType type = DefaultArbitration ) const;
     float segmentXErr    ( int station, int muonSubdetId, ArbitrationType type = DefaultArbitration ) const;
     float segmentYErr    ( int station, int muonSubdetId, ArbitrationType type = DefaultArbitration ) const;
     float segmentDxDzErr ( int station, int muonSubdetId, ArbitrationType type = DefaultArbitration ) const;
     float segmentDyDzErr ( int station, int muonSubdetId, ArbitrationType type = DefaultArbitration ) const;
     /// get track information in chamber
     /// If no chamber returns 999999
     float trackEdgeX   ( int station, int muonSubdetId ) const;
     float trackEdgeY   ( int station, int muonSubdetId ) const;
     float trackX       ( int station, int muonSubdetId ) const;
     float trackY       ( int station, int muonSubdetId ) const;
     float trackDxDz    ( int station, int muonSubdetId ) const;
     float trackDyDz    ( int station, int muonSubdetId ) const;
     float trackXErr    ( int station, int muonSubdetId ) const;
     float trackYErr    ( int station, int muonSubdetId ) const;
     float trackDxDzErr ( int station, int muonSubdetId ) const;
     float trackDyDzErr ( int station, int muonSubdetId ) const;

  };

}

#include "DataFormats/MuonReco/interface/MuonFwd.h"

#endif
