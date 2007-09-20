//
// $Id$
//

#ifndef TopObjects_TopElectron_h
#define TopObjects_TopElectron_h

/**
  \class    TopElectron TopElectron.h "AnalysisDataFormats/TopObjects/interface/TopElectron.h"
  \brief    High-level top electron container

   TopElectron contains an electron as a TopObject, and provides the means to
   store and retrieve the high-level additional information.

  \author   Steven Lowette
  \version  $Id$
*/

#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "AnalysisDataFormats/TopObjects/interface/TopLepton.h"


typedef reco::PixelMatchGsfElectron TopElectronType;
typedef reco::PixelMatchGsfElectronCollection TopElectronTypeCollection;


class TopElectron : public TopLepton<TopElectronType> {

  friend class TopElectronProducer;

  public:

    TopElectron();
    TopElectron(const TopElectronType & anElectron);
    virtual ~TopElectron();
  
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
