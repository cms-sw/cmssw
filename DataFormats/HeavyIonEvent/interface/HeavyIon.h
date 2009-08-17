#ifndef DataFormats_HeavyIon_h 
#define DataFormats_HeavyIon_h

#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"

namespace pat {

   class HeavyIon : public reco::Centrality, public reco::EvtPlane {
   public:
      HeavyIon();
      HeavyIon(const reco::Centrality & c, const reco::EvtPlane & e);
      HeavyIon(const reco::Centrality & c, const reco::EvtPlane & e, double b, int npart, int ncoll, int nhard, double phi);
      virtual ~HeavyIon(){;}

      bool initializeCentrality();

      bool isMC(){return isMC_;}
      double generatedB() {return b_;}
      int generatedNpart() {return npart_;}
      int generatedNcoll() {return ncoll_;}
      int generatedNhard() {return nhard_;}
      double generatedEvtPlane() {return phi_;}

      int centralityBin();

   private:
      bool isMC_;
      double b_;
      int npart_;
      int ncoll_;
      int nhard_;
      double phi_;

   };

}

#endif
