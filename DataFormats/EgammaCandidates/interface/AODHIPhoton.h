#ifndef AODHIPhoton_h
#define AODHIPhoton_h

#include "DataFormats/EgammaCandidates/interface/Photon.h"

namespace aod{

  class AODHIPhoton : public reco::Photon {

  public:

  AODHIPhoton() : reco::Photon() {}
    virtual ~AODHIPhoton() {}

  AODHIPhoton(const reco::Photon& p) : reco::Photon(p) {}

    float cc1() {return cc1_;}
    float cc2() {return cc2_;}
    float cc3() {return cc3_;}
    float cc4() {return cc4_;}
    float cc5() {return cc5_;}

    float cr1() {return cr1_;}
    float cr2() {return cr2_;}
    float cr3() {return cr3_;}
    float cr4() {return cr4_;}
    float cr5() {return cr5_;}

    float ct1PtCut20() {return ct1PtCut20_;}
    float ct2PtCut20() {return ct2PtCut20_;}
    float ct3PtCut20() {return ct3PtCut20_;}
    float ct4PtCut20() {return ct4PtCut20_;}
    float ct5PtCut20() {return ct5PtCut20_;}

    float swissCrx() {return swissCrx_;}
    float seedTime() {return seedTime_;}

    void setcc1(float cc1) {cc1_ = cc1;}
    void setcc2(float cc2) {cc2_ = cc2;}
    void setcc3(float cc3) {cc3_ = cc3;}
    void setcc4(float cc4) {cc4_ = cc4;}
    void setcc5(float cc5) {cc5_ = cc5;}

    void setcr1(float cr1) {cr1_ = cr1;}
    void setcr2(float cr2) {cr2_ = cr2;}
    void setcr3(float cr3) {cr3_ = cr3;}
    void setcr4(float cr4) {cr4_ = cr4;}
    void setcr5(float cr5) {cr5_ = cr5;}

    void setct1PtCut20(float ct1PtCut20) {ct1PtCut20_ = ct1PtCut20;}
    void setct2PtCut20(float ct2PtCut20) {ct2PtCut20_ = ct2PtCut20;}
    void setct3PtCut20(float ct3PtCut20) {ct3PtCut20_ = ct3PtCut20;}
    void setct4PtCut20(float ct4PtCut20) {ct4PtCut20_ = ct4PtCut20;}
    void setct5PtCut20(float ct5PtCut20) {ct5PtCut20_ = ct5PtCut20;}

    void swissCrx(float swissCrx) {swissCrx_ = swissCrx;}
    void seedTime(float seedTime) {seedTime_ = seedTime;}


  private:

    float cc1_, cc2_, cc3_, cc4_, cc5_;
    float cr1_, cr2_, cr3_, cr4_, cr5_;
    float ct1PtCut20_, ct2PtCut20_, ct3PtCut20_, ct4PtCut20_, ct5PtCut20_;

    float swissCrx_, seedTime_;
  };

  typedef std::vector<AODHIPhoton> AODHIPhotonCollection;

}
#endif
