#ifndef DataFormats_TauReco_BaseTau_h
#define DataFormats_TauReco_BaseTau_h

/* class BaseTau
 * base class 
 * authors: Simone Gennai (simone.gennai@cern.ch), Ludovic Houchu (Ludovic.Houchu@cern.ch)
 * created: Jun 21 2007,
 * revised: Sep 4 2007
 */
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TauReco/interface/BaseTauFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <limits>


namespace reco {
  class BaseTau : public RecoCandidate {
  public:
    BaseTau();
    BaseTau(Charge q,const LorentzVector &,const Point & = Point(0,0,0));
    virtual ~BaseTau(){}
    BaseTau* clone()const;

    // rec. jet Lorentz-vector combining (Tracks and neutral ECAL Island BasicClusters) or (charged hadr. PFCandidates and gamma PFCandidates)
    math::XYZTLorentzVector alternatLorentzVect()const;
    void setalternatLorentzVect(const math::XYZTLorentzVector&);
    
    // leading Track
    virtual reco::TrackRef leadTrack() const;
    void setleadTrack(const TrackRef&);
    
    // Tracks which passed quality cuts and are inside a tracker signal cone around leading Track 
    virtual const reco::TrackRefVector& signalTracks() const;
    void setsignalTracks(const TrackRefVector&);
 
    // Tracks which passed quality cuts and are inside a tracker isolation annulus around leading Track 
    virtual const reco::TrackRefVector& isolationTracks() const;
    void setisolationTracks(const TrackRefVector&);  
  private:
    // check overlap with another candidate
    virtual bool overlap(const Candidate&)const;
    math::XYZTLorentzVector alternatLorentzVect_;
    reco::TrackRef leadTrack_;
    reco::TrackRefVector signalTracks_, isolationTracks_;
  };
}
#endif
