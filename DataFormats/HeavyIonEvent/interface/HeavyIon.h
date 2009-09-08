#ifndef DataFormats_HeavyIon_h 
#define DataFormats_HeavyIon_h

#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "DataFormats/HeavyIonEvent/interface/EvtPlane.h"

namespace pat {

   class HeavyIon {
   public:
      HeavyIon();
      HeavyIon(const reco::CentralityCollection & c, const reco::EvtPlaneCollection & e);
      HeavyIon(const reco::CentralityCollection & c, const reco::EvtPlaneCollection & e, double b, int npart, int ncoll, int nhard, double phi);
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
      reco::CentralityCollection cents_;
      reco::EvtPlaneCollection planes_;
      bool isMC_;
      double b_;
      int npart_;
      int ncoll_;
      int nhard_;
      double phi_;
   };

}

#endif
