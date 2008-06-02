//
// $Id: Tau.h,v 1.9.2.2 2008/05/14 13:20:38 lowette Exp $
//

#ifndef DataFormats_PatCandidates_Tau_h
#define DataFormats_PatCandidates_Tau_h

/**
  \class    pat::Tau Tau.h "DataFormats/PatCandidates/interface/Tau.h"
  \brief    Analysis-level tau class

   Tau implements the analysis-level tau class within the 'pat' namespace.

  \author   Steven Lowette
  \version  $Id: Tau.h,v 1.9.2.2 2008/05/14 13:20:38 lowette Exp $
*/


#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/PatCandidates/interface/Lepton.h"


namespace pat {


  typedef reco::BaseTau TauType;


  class Tau : public Lepton<TauType> {

    public:

      Tau();
      Tau(const TauType & aTau);
      Tau(const edm::RefToBase<TauType> & aTauRef);
      virtual ~Tau();

      virtual Tau * clone() const { return new Tau(*this); }

      /// override the TauType::isolationTracks method, to access the internal storage of the track
      const reco::TrackRefVector & isolationTracks() const;
      /// override the TauType::track method, to access the internal storage of the track
      const reco::TrackRef & leadTrack() const;
      /// override the TauType::track method, to access the internal storage of the track
      const reco::TrackRefVector & signalTracks() const;
      float emEnergyFraction() const { return emEnergyFraction_; }
      float eOverP() const { return eOverP_; }
      float leadEoverP() const { return leadeOverP_; }
      float hHotOverP() const { return HhotOverP_; }
      float hTotOverP() const { return HtotOverP_; }

      /// method to store the isolation tracks internally
      void embedIsolationTracks();
      /// method to store the isolation tracks internally
      void embedLeadTrack();
      /// method to store the isolation tracks internally
      void embedSignalTracks();
      void setEmEnergyFraction(float fraction) { emEnergyFraction_ = fraction; }
      void setEOverP(float EoP) { eOverP_ = EoP; } 
      void setLeadEOverP(float EoP) { leadeOverP_ = EoP; }
      void setHhotOverP(float HHoP) { HhotOverP_ = HHoP; }
      void setHtotOverP(float HToP) { HtotOverP_ = HToP; }

    private:

      bool embeddedIsolationTracks_;
      std::vector<reco::Track> isolationTracks_;
      mutable TrackRefVector transientIsolationTracksRefVector_;
      bool embeddedLeadTrack_;
      std::vector<reco::Track> leadTrack_;
      mutable TrackRef transientLeadTrackRef_;
      bool embeddedSignalTracks_;
      std::vector<reco::Track> signalTracks_;
      mutable TrackRefVector transientSignalTracksRefVector_;
      float emEnergyFraction_;
      float eOverP_;
      float leadeOverP_;
      float HhotOverP_;
      float HtotOverP_;

  };


}

#endif
