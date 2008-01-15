//
// $Id: Muon.h,v 1.1 2008/01/07 11:48:25 lowette Exp $
//

#ifndef DataFormats_PatCandidates_Muon_h
#define DataFormats_PatCandidates_Muon_h

/**
  \class    Muon Muon.h "DataFormats/PatCandidates/interface/Muon.h"
  \brief    Analysis-level muon class

   Muon implements the analysis-level muon class within the 'pat' namespace.

  \author   Steven Lowette
  \version  $Id: Muon.h,v 1.1 2008/01/07 11:48:25 lowette Exp $
*/

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "DataFormats/PatCandidates/interface/Lepton.h"


namespace pat {


  typedef reco::Muon MuonType;
  typedef reco::MuonCollection MuonTypeCollection;


  class Muon : public Lepton<MuonType> {

    friend class PATMuonProducer;

    public:

      Muon();
      Muon(const MuonType & aMuon);
      virtual ~Muon();

      float getTrackIso() const;
      float getCaloIso() const;
      float getLeptonID() const;

    protected:

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
