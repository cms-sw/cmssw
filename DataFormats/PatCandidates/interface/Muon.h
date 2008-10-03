//
// $Id: Muon.h,v 1.15 2008/06/17 13:50:56 gpetrucc Exp $
//

#ifndef DataFormats_PatCandidates_Muon_h
#define DataFormats_PatCandidates_Muon_h

/**
  \class    pat::Muon Muon.h "DataFormats/PatCandidates/interface/Muon.h"
  \brief    Analysis-level muon class

   Muon implements the analysis-level muon class within the 'pat' namespace.

  \author   Steven Lowette
  \version  $Id: Muon.h,v 1.15 2008/06/17 13:50:56 gpetrucc Exp $
*/

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/PatCandidates/interface/Lepton.h"

#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidate.h"

namespace pat {


  typedef reco::Muon MuonType;
  typedef reco::MuonCollection MuonTypeCollection;


  class Muon : public Lepton<MuonType> {

    public:

      /// default constructor
      Muon();
      /// constructor from MuonType
      Muon(const MuonType & aMuon);
      /// constructor from ref to MuonType
      Muon(const edm::RefToBase<MuonType> & aMuonRef);
      /// constructor from ref to MuonType
      Muon(const edm::Ptr<MuonType> & aMuonRef);
      /// destructor
      virtual ~Muon();

      /// reimplementation of the Candidate clone method
      virtual Muon * clone() const { return new Muon(*this); }

      /// add a reference to the source IsolatedPFCandidate
      void setPFCandidateRef(const reco::IsolatedPFCandidateRef& ref) {
	pfCandidateRef_ = ref;
      } 

      /// reference to Track reconstructed in the tracker only (reimplemented from reco::Muon)
      reco::TrackRef track() const;
      /// reference to Track reconstructed in the tracker only (reimplemented from reco::Muon)
      reco::TrackRef innerTrack() const { return track(); }

      /// reference to Track reconstructed in the muon detector only (reimplemented from reco::Muon)
      reco::TrackRef standAloneMuon() const;
      /// reference to Track reconstructed in the muon detector only (reimplemented from reco::Muon)
      reco::TrackRef outerTrack() const { return standAloneMuon(); }

      /// reference to Track reconstructed in both tracked and muon detector (reimplemented from reco::Muon)
      reco::TrackRef combinedMuon() const;
      /// reference to Track reconstructed in both tracked and muon detector (reimplemented from reco::Muon)
      reco::TrackRef globalTrack() const { return combinedMuon(); }

      /// reference to the source IsolatedPFCandidates
      /// null if this has been built from a standard muon
      reco::IsolatedPFCandidateRef pfCandidateRef() const;

      /// return the lepton ID discriminator
      float leptonID() const;
      /// return whether it is a good muon
      bool isGoodMuon(const MuonType & muon, reco::Muon::SelectionType type = reco::Muon::TMLastStationLoose);
      /// return the muon segment compatibility -> meant for
      float segmentCompatibility() const;


      /// set reference to Track reconstructed in the tracker only (reimplemented from reco::Muon)
      void embedTrack();
      /// set reference to Track reconstructed in the muon detector only (reimplemented from reco::Muon)
      void embedStandAloneMuon();
      /// set reference to Track reconstructed in both tracked and muon detector (reimplemented from reco::Muon)
      void embedCombinedMuon();

      /// embed the IsolatedPFCandidate pointed to by pfCandidateRef_
      void embedPFCandidate();

      /// method to set the lepton ID discriminator
      void setLeptonID(float id);

    protected:

      bool embeddedTrack_;
      std::vector<reco::Track> track_;
      bool embeddedStandAloneMuon_;
      std::vector<reco::Track> standAloneMuon_;
      bool embeddedCombinedMuon_;
      std::vector<reco::Track> combinedMuon_;

      /// true if the IsolatedPFCandidate is embedded
      bool embeddedPFCandidate_;      

      /// if embeddedPFCandidate_, a copy of the source IsolatedPFCandidate
      /// is stored in this vector
      reco::IsolatedPFCandidateCollection pfCandidate_;

      /// reference to the IsolatedPFCandidate this has been built from
      /// null if this has been built from a standard muon
      reco::IsolatedPFCandidateRef        pfCandidateRef_;

      float leptonID_;

  };


}

#endif
