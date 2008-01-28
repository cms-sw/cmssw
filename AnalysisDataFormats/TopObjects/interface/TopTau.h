#ifndef TopObjects_TopTau_h
#define TopObjects_TopTau_h

/**
  \class    TopTau TopTau.h "AnalysisDataFormats/TopObjects/interface/TopTau.h"
  \brief    High-level top tau container

   TopTau contains a tau as a TopObject, and provides the means to
   store and retrieve the high-level additional information.

  \author   Christophe Delaere
  \version  $Id: TopTau.h,v 1.3 2007/10/30 09:59:05 delaer Exp $
*/

#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/BTauReco/interface/IsolatedTauTagInfo.h"

#include "AnalysisDataFormats/TopObjects/interface/TopLepton.h"

typedef reco::BaseTau TopTauType;

class TopTau : public TopLepton<TopTauType> {

  friend class TopTauProducer;

  public:

    TopTau();
    TopTau(const TopTauType&);
    virtual ~TopTau();

    double getEmEnergyFraction() const { return emEnergyFraction_; }
    double getEoverP() const { return eOverP_; }
    double getHhotOverP() const { return HhotOverP_; }
    double getEcalIsolation() const { return ecalIsolation_; }
    double getHtotOverP() const { return HtotOverP_; }
  
  protected: 
    
    void setEmEnergyFraction(double fraction) { emEnergyFraction_ = fraction; }
    void setEoverP(double EoP) { eOverP_ = EoP; } 
    void setHhotOverP(double HHoP) {HhotOverP_ = HHoP; }
    void setEcalIsolation(double Eisol) { ecalIsolation_ = Eisol; }
    void setHtotOverP(double HToP) { HtotOverP_ = HToP; }
  
  private:

    double emEnergyFraction_;
    double eOverP_;
    double HhotOverP_;
    double ecalIsolation_;
    double HtotOverP_;
  
};

#endif
