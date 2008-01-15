//
// $Id: Tau.h,v 1.1 2008/01/07 11:48:25 lowette Exp $
//

#ifndef DataFormats_PatCandidates_Tau_h
#define DataFormats_PatCandidates_Tau_h

/**
  \class    Tau Tau.h "DataFormats/PatCandidates/interface/Tau.h"
  \brief    Analysis-level tau class

   Tau implements the analysis-level tau class within the 'pat' namespace.

  \author   Steven Lowette
  \version  $Id: Tau.h,v 1.1 2008/01/07 11:48:25 lowette Exp $
*/

#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/PatCandidates/interface/Lepton.h"


namespace pat {


  typedef reco::BaseTau TauType;


  class Tau : public Lepton<TauType> {

    friend class PATTauProducer;

    public:

      Tau();
      Tau(const TauType&);
      virtual ~Tau();

      float getEmEnergyFraction() const { return emEnergyFraction_; }
      float getEoverP() const { return eOverP_; }

    protected: 
    
      void setEmEnergyFraction(float fraction) { emEnergyFraction_ = fraction; }
      void setEoverP(float EoP) { eOverP_ = EoP; } 

    private:

      float emEnergyFraction_;
      float eOverP_;

  };


}

#endif
