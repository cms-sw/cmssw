#ifndef HcalIsolatedTrack_IsolatedPixelTrackCandidate_h
#define HcalIsolatedTrack_IsolatedPixelTrackCandidate_h
/** \class reco::IsolatedPixelTrackCandidate
 *
 *
 */

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"

namespace reco {

  class IsolatedPixelTrackCandidate: public RecoCandidate {
    
  public:

        /// default constructor
    IsolatedPixelTrackCandidate() : RecoCandidate() { }
    /// constructor from a track
      IsolatedPixelTrackCandidate(const reco::TrackRef & tr, bool tjbit, bool jbit, double max, double sum): 
      RecoCandidate( 0, LorentzVector(tr->px(),tr->py(),tr->pz(),tr->p()) ),
	track_(tr),l1taujetBit_(tjbit), l1jetBit_(jbit), maxPtPxl_(max),sumPtPxl_(sum) {}
    /// destructor
    virtual ~IsolatedPixelTrackCandidate();
    /// returns a clone of the candidate
    virtual IsolatedPixelTrackCandidate * clone() const;
    /// refrence to a Track
    virtual reco::TrackRef track() const;
    /// get decision of L1_SingleTauJet trigger
    bool l1taujetRes() const {return l1taujetBit_;}
    /// get decision of L1_SingleJet trigger
    bool l1jetRes() const {return l1jetBit_;}
    /// highest Pt of other pixel tracks in the cone around the candidate
    double maxPtPxl() const {return maxPtPxl_;}
    /// Pt sum of other pixel tracks in the cone around the candidate
    double sumPtPxl() const {return sumPtPxl_;}
    
    /// set refrence to Track component
    void setTrack( const reco::TrackRef & tr ) { track_ = tr; }

  private:
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
    /// reference to a Track
    reco::TrackRef track_;
    /// decision of L1_SingleTauJet trigger
    bool l1taujetBit_;
    ///decision of L1_SingleJet trigger
    bool l1jetBit_;
    /// highest Pt of other pixel tracks in the cone around the candidate
    double maxPtPxl_;
    /// Pt sum of other pixel tracks in the cone around the candidate
    double sumPtPxl_;
    
  };


}

#endif
