#ifndef HcalIsolatedTrack_IsolatedPixelTrackCandidate_h
#define HcalIsolatedTrack_IsolatedPixelTrackCandidate_h
/** \class reco::IsolatedPixelTrackCandidate
 *
 *
 */

#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

#include "DataFormats/HcalIsolatedTrack/interface/IsolatedPixelTrackCandidateFwd.h"

#include <vector>
#include <map>
#include <utility>

namespace reco {
  
  class IsolatedPixelTrackCandidate: public RecoCandidate {
    
  public:
    
    /// default constructor
    IsolatedPixelTrackCandidate() : RecoCandidate() { }
      ///constructor from LorentzVector
      IsolatedPixelTrackCandidate(const LorentzVector& v): RecoCandidate(0,v)
	{
	  enIn_=-1;
	  enOut_=-1;
	  nhitIn_=-1;
	  nhitOut_=-1;
	  maxPtPxl_=-1;
	  sumPtPxl_=-1;
	}
      /// constructor from a track
      IsolatedPixelTrackCandidate(const reco::TrackRef & tr, const l1extra::L1JetParticleRef & tauRef, double max, double sum): 
	RecoCandidate( 0, LorentzVector((tr.get()->px()),(tr.get())->py(),(tr.get())->pz(),(tr.get())->p()) ),
	track_(tr), l1tauJet_(tauRef), maxPtPxl_(max), sumPtPxl_(sum) 
	{
	  enIn_=-1;
	  enOut_=-1;
	  nhitIn_=-1;
	  nhitOut_=-1;
	}
	
	///constructor from tau jet
	IsolatedPixelTrackCandidate(const l1extra::L1JetParticleRef & tauRef, double enIn, double enOut, int nhitIn, int nhitOut):
	  RecoCandidate( 0, LorentzVector(tauRef->px(),tauRef->py(),tauRef->pz(),tauRef->p()) ), 
	  l1tauJet_(tauRef), enIn_(enIn), enOut_(enOut), nhitIn_(nhitIn), nhitOut_(nhitOut) 
	  {
	    maxPtPxl_=-1;
	    sumPtPxl_=-1;
	  }
	  
	  
	  
	  
	  /// destructor
	  virtual ~IsolatedPixelTrackCandidate();

	  /// returns a clone of the candidate
	  virtual IsolatedPixelTrackCandidate * clone() const;

	  /// refrence to a Track
	  virtual reco::TrackRef track() const;
          void setTrack( const reco::TrackRef & tr ) { track_ = tr; }

	  /// highest Pt of other pixel tracks in the cone around the candidate
	  double maxPtPxl() const {return maxPtPxl_;}
	  void SetMaxPtPxl(double mptpxl) {maxPtPxl_=mptpxl;}

	  /// Pt sum of other pixel tracks in the cone around the candidate
	  double sumPtPxl() const {return sumPtPxl_;}
	  void SetSumPtPxl(double sumptpxl) {sumPtPxl_=sumptpxl;}
	  
	  /// get reference to L1 tau jet
	  virtual l1extra::L1JetParticleRef l1tau() const;
          void setL1TauJet( const l1extra::L1JetParticleRef & tauRef ) { l1tauJet_ = tauRef; }
 	  
	  /// ECAL energy in the inner cone around tau jet
	  double energyIn() const {return enIn_; }
	  void SetEnergyIn(double a) {enIn_=a;}
	  
	  /// ECAL energy in the outer cone around tau jet
	  double energyOut() const {return enOut_;}
	  void SetEnergyOut(double a) {enOut_=a;}
	  
	  /// number of ECAL hits in the inner cone around tau jet
	  int nHitIn() const {return nhitIn_;}
	  void SetNHitIn(int a) {nhitIn_=a;}
	  
	  /// number of ECAL hits in the outer cone around tau jet
	  int nHitOut() const {return nhitOut_;}
	  void SetNHitOut(int a) {nhitOut_=a;}
	  
	  ///get index of tower which track is hitting
	  std::pair<int,int> towerIndex() const;
	  
  private:
	  /// check overlap with another candidate
	  virtual bool overlap( const Candidate & ) const;
	  /// reference to a Track
	  reco::TrackRef track_;
	  /// reference to a L1 tau jet
	  l1extra::L1JetParticleRef l1tauJet_;
	  /// highest Pt of other pixel tracks in the cone around the candidate
	  double maxPtPxl_;
	  /// Pt sum of other pixel tracks in the cone around the candidate
	  double sumPtPxl_;
	  /// energy in inner cone around L1 tau jet
	  double enIn_;
	  /// energy in outer cone around L1 tau jet
	  double enOut_;
	  /// number of hits in inner cone
	  int nhitIn_;
	  /// number of hits in inner cone
	  int nhitOut_;
	  
  };
  
  
}

#endif
