//
// $Id: TopMuon.h,v 1.1 2007/09/20 18:12:22 lowette Exp $
//

#ifndef TopObjects_TopMuon_h
#define TopObjects_TopMuon_h

/**
  \class    TopMuon TopMuon.h "AnalysisDataFormats/TopObjects/interface/TopMuon.h"
  \brief    High-level muon container

   TopMuon contains a muon as a TopObject, and provides the means to
   store and retrieve the high-level additional information.

  \author   Steven Lowette
  \version  $Id: TopMuon.h,v 1.1 2007/09/20 18:12:22 lowette Exp $
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

    void setTrackIso(double trackIso);
    void setCaloIso(double caloIso);
    void setLeptonID(double id);

  protected:

    double trackIso_;
    double caloIso_;
    double leptonID_;

};


#endif
