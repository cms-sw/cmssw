//
// $Id: Tau.h,v 1.2 2008/01/16 20:33:24 lowette Exp $
//

#ifndef DataFormats_PatCandidates_Tau_h
#define DataFormats_PatCandidates_Tau_h

/**
  \class    Tau Tau.h "DataFormats/PatCandidates/interface/Tau.h"
  \brief    Analysis-level tau class

   Tau implements the analysis-level tau class within the 'pat' namespace.

  \author   Steven Lowette
  \version  $Id: Tau.h,v 1.2 2008/01/16 20:33:24 lowette Exp $
*/

#include "DataFormats/TauReco/interface/Tau.h"
/* > 1.8.X functionality: #include "DataFormats/TauReco/interface/BaseTau.h" */
#include "DataFormats/PatCandidates/interface/Lepton.h"


namespace pat {


  typedef reco::Tau TauType;
  /* > 1.8.X typedef reco::BaseTau TauType; */


  class Tau : public Lepton<TauType> {

    public:

      Tau();
      Tau(const TauType&);
      virtual ~Tau();

      float emEnergyFraction() const { return emEnergyFraction_; }
      float eOverP() const { return eOverP_; }

      void setEmEnergyFraction(float fraction) { emEnergyFraction_ = fraction; }
      void setEOverP(float EoP) { eOverP_ = EoP; } 

    private:

      float emEnergyFraction_;
      float eOverP_;

  };


}

#endif
