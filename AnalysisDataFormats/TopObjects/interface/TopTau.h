#ifndef TopObjects_TopTau_h
#define TopObjects_TopTau_h

/**
  \class    TopTau TopTau.h "AnalysisDataFormats/TopObjects/interface/TopTau.h"
  \brief    High-level top tau container

   TopTau contains a tau as a TopObject, and provides the means to
   store and retrieve the high-level additional information.

  \author   Christophe Delaere
  \version  $Id: TopTau.h,v 1.2 2007/10/04 15:41:54 delaer Exp $
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
  
  protected: 
    
    void setEmEnergyFraction(double fraction) { emEnergyFraction_ = fraction; }
    void setEoverP(double EoP) { eOverP_ = EoP; } 
  
  private:

    double emEnergyFraction_;
    double eOverP_;
  
};

#endif
