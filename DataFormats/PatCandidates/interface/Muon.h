//
// $Id: Muon.h,v 1.6 2008/02/11 15:53:49 llista Exp $
//

#ifndef DataFormats_PatCandidates_Muon_h
#define DataFormats_PatCandidates_Muon_h

/**
  \class    pat::Muon Muon.h "DataFormats/PatCandidates/interface/Muon.h"
  \brief    Analysis-level muon class

   Muon implements the analysis-level muon class within the 'pat' namespace.

  \author   Steven Lowette
  \version  $Id: Muon.h,v 1.6 2008/02/11 15:53:49 llista Exp $
*/

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/PatCandidates/interface/Lepton.h"


namespace pat {


  typedef reco::Muon MuonType;
  typedef reco::MuonCollection MuonTypeCollection;


  class Muon : public Lepton<MuonType> {

    public:

      Muon();
      Muon(const MuonType & aMuon);
      Muon(const edm::RefToBase<MuonType> & aMuonRef);
      virtual ~Muon();

      float trackIso() const;
      float caloIso() const;
      float leptonID() const;

      void setTrackIso(float trackIso);
      void setCaloIso(float caloIso);
      void setLeptonID(float id);

    protected:

      float trackIso_;
      float caloIso_;
      float leptonID_;

  };


}

#endif
