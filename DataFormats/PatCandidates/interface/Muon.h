//
// $Id: Muon.h,v 1.9 2008/04/04 18:22:08 srappocc Exp $
//

#ifndef DataFormats_PatCandidates_Muon_h
#define DataFormats_PatCandidates_Muon_h

/**
  \class    pat::Muon Muon.h "DataFormats/PatCandidates/interface/Muon.h"
  \brief    Analysis-level muon class

   Muon implements the analysis-level muon class within the 'pat' namespace.

  \author   Steven Lowette
  \version  $Id: Muon.h,v 1.9 2008/04/04 18:22:08 srappocc Exp $
*/

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/PatCandidates/interface/Lepton.h"

#include "RecoMuon/MuonIdentification/interface/IdGlobalFunctions.h"


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

      /// return the lepton ID discriminator
      float leptonID() const;
      /// return whether it is a good muon
      bool isGoodMuon(const MuonType & muon, muonid::SelectionType type = muonid::TMLastStationLoose);
      /// return the muon segment compatibility -> meant for
      float segmentCompatibility() const;

      /// method to set the lepton ID discriminator
      void setLeptonID(float id);

    protected:

      float leptonID_;

  };


}

#endif
