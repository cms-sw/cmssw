//
// $Id: Muon.h,v 1.8.2.5 2008/05/14 13:20:38 lowette Exp $
//

#ifndef DataFormats_PatCandidates_Muon_h
#define DataFormats_PatCandidates_Muon_h

/**
  \class    pat::Muon Muon.h "DataFormats/PatCandidates/interface/Muon.h"
  \brief    Analysis-level muon class

   Muon implements the analysis-level muon class within the 'pat' namespace.

  \author   Steven Lowette
  \version  $Id: Muon.h,v 1.8.2.5 2008/05/14 13:20:38 lowette Exp $
*/

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/PatCandidates/interface/Lepton.h"

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
      /// destructor
      virtual ~Muon();

      /// reimplementation of the Candidate clone method
      virtual Muon * clone() const { return new Muon(*this); }

      /// reference to Track reconstructed in the tracker only (reimplemented from reco::Muon)
      reco::TrackRef track() const;
      /// reference to Track reconstructed in the muon detector only (reimplemented from reco::Muon)
      reco::TrackRef standAloneMuon() const;
      /// reference to Track reconstructed in both tracked and muon detector (reimplemented from reco::Muon)
      reco::TrackRef combinedMuon() const;
      /// return the lepton ID discriminator
      float leptonID() const;
      /// return the muon segment compatibility
      float segmentCompatibility() const;

      /// set reference to Track reconstructed in the tracker only (reimplemented from reco::Muon)
      void embedTrack();
      /// set reference to Track reconstructed in the muon detector only (reimplemented from reco::Muon)
      void embedStandAloneMuon();
      /// set reference to Track reconstructed in both tracked and muon detector (reimplemented from reco::Muon)
      void embedCombinedMuon();
      /// method to set the lepton ID discriminator
      void setLeptonID(float id);

    protected:

      bool embeddedTrack_;
      std::vector<reco::Track> track_;
      bool embeddedStandAloneMuon_;
      std::vector<reco::Track> standAloneMuon_;
      bool embeddedCombinedMuon_;
      std::vector<reco::Track> combinedMuon_;
      float leptonID_;

  };


}

#endif
