//
// $Id: Tau.h,v 1.8 2008/02/21 21:23:27 delaer Exp $
//

#ifndef DataFormats_PatCandidates_Tau_h
#define DataFormats_PatCandidates_Tau_h

/**
  \class    pat::Tau Tau.h "DataFormats/PatCandidates/interface/Tau.h"
  \brief    Analysis-level tau class

   Tau implements the analysis-level tau class within the 'pat' namespace.

  \author   Steven Lowette
  \version  $Id: Tau.h,v 1.8 2008/02/21 21:23:27 delaer Exp $
*/


#include "DataFormats/TauReco/interface/BaseTau.h"
#include "DataFormats/PatCandidates/interface/Lepton.h"


namespace pat {


  typedef reco::BaseTau TauType;


  class Tau : public Lepton<TauType> {

    public:

      Tau();
      Tau(const TauType & aTau);
      Tau(const edm::RefToBase<TauType> & aTauRef);
      virtual ~Tau();

      float emEnergyFraction() const { return emEnergyFraction_; }
      float eOverP() const { return eOverP_; }
      float leadEoverP() const { return leadeOverP_; }
      float hHotOverP() const { return HhotOverP_; }
      float hTotOverP() const { return HtotOverP_; }

      void setEmEnergyFraction(float fraction) { emEnergyFraction_ = fraction; }
      void setEOverP(float EoP) { eOverP_ = EoP; } 
      void setLeadEOverP(float EoP) { leadeOverP_ = EoP; }
      void setHhotOverP(float HHoP) {HhotOverP_ = HHoP; }
      void setHtotOverP(float HToP) { HtotOverP_ = HToP; }

    private:

      float emEnergyFraction_;
      float eOverP_;
      float leadeOverP_;
      float HhotOverP_;
      float HtotOverP_;

  };


}

#endif
