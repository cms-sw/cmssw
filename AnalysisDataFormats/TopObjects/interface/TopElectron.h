//
// $Id: TopElectron.h,v 1.4 2007/12/14 13:54:25 jlamb Exp $
//

#ifndef TopObjects_TopElectron_h
#define TopObjects_TopElectron_h

/**
  \class    TopElectron TopElectron.h "AnalysisDataFormats/TopObjects/interface/TopElectron.h"
  \brief    High-level top electron container

   TopElectron contains an electron as a TopObject, and provides the means to
   store and retrieve the high-level additional information.

  \author   Steven Lowette
  \version  $Id: TopElectron.h,v 1.4 2007/12/14 13:54:25 jlamb Exp $
*/

#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
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
    double getElectronIDRobust() const;
    double getEgammaTkIso() const;
    int getEgammaTkNumIso() const;
    double getEgammaEcalIso() const;
    double getEgammaHcalIso() const;


  protected:

    void setTrackIso(double trackIso);
    void setCaloIso(double caloIso);
    void setLeptonID(double id);
    void setElectronIDRobust(double id);
    void setEgammaTkIso(double tkIso);
    void setEgammaTkNumIso(int tkNumIso);
    void setEgammaEcalIso(double ecalIso);
    void setEgammaHcalIso(double hcalIso);

    

  protected:

    double trackIso_;
    double caloIso_;
    double leptonID_;
    double electronIDRobust_;
    double egammaTkIso_;
    int egammaTkNumIso_;
    double egammaEcalIso_;
    double egammaHcalIso_;
};


#endif
