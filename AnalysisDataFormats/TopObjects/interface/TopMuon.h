//
// $Id$
//

#ifndef TopObjects_TopMuon_h
#define TopObjects_TopMuon_h

/**
  \class    TopMuon TopMuon.h "AnalysisDataFormats/TopObjects/interface/TopMuon.h"
  \brief    High-level muon container

   TopMuon contains a muon as a TopObject, and provides the means to
   store and retrieve the high-level additional information.

  \author   Steven Lowette
  \version  $Id$
*/

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "AnalysisDataFormats/TopObjects/interface/TopLepton.h"


typedef reco::Muon TopMuonType;
typedef reco::MuonCollection TopMuonTypeCollection;


class TopMuon : public TopLepton<TopMuonType> {

  friend class TopMuonProducer;

  public:

    TopMuon();
    TopMuon(const TopMuonType & aMuon);
    virtual ~TopMuon();

    double getTrackIso() const;
    double getCaloIso() const;
    double getLeptonID() const;

  protected:

    void setLeptonID(double id);

  protected:

    double leptonID_;

};


#endif
