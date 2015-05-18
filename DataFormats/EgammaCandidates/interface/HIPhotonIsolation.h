#ifndef EgammaCandidates_HIPhotonIsolation_h
#define EgammaCandidates_HIPhotonIsolation_h

#include "DataFormats/Common/interface/ValueMap.h"

namespace reco{

  class HIPhotonIsolation {

  public:

  HIPhotonIsolation() : cc1_(0),
      cc2_(0),
      cc3_(0),
      cc4_(0),
      cc5_(0),
      cr1_(0),
      cr2_(0),
      cr3_(0),
      cr4_(0),
      cr5_(0),
      ct1PtCut20_(0),
      ct2PtCut20_(0),
      ct3PtCut20_(0),
      ct4PtCut20_(0),
      ct5PtCut20_(0),
      swissCrx_(0),
      seedTime_(0)
      {}
    virtual ~HIPhotonIsolation() {}

    //getters

    /// Cluster-based isolation (ECAL) R = 0.1
    float cc1() const {return cc1_;}
    /// Cluster-based isolation (ECAL) R = 0.2
    float cc2() const {return cc2_;}
    /// Cluster-based isolation (ECAL) R = 0.3
    float cc3() const {return cc3_;}
    /// Cluster-based isolation (ECAL) R = 0.4
    float cc4() const {return cc4_;}
    /// Cluster-based isolation (ECAL) R = 0.5
    float cc5() const {return cc5_;}

    /// Rechit-based isolation (HCAL) R = 0.1
    float cr1() const {return cr1_;}
    /// Rechit-based isolation (HCAL) R = 0.2
    float cr2() const {return cr2_;}
    /// Rechit-based isolation (HCAL) R = 0.3
    float cr3() const {return cr3_;}
    /// Rechit-based isolation (HCAL) R = 0.4
    float cr4() const {return cr4_;}
    /// Rechit-based isolation (HCAL) R = 0.5
    float cr5() const {return cr5_;}

    /// Track-based isolation, pt>2.0GeV, R = 0.1
    float ct1PtCut20() const {return ct1PtCut20_;}
    /// Track-based isolation, pt>2.0GeV, R = 0.2
    float ct2PtCut20() const {return ct2PtCut20_;}
    /// Track-based isolation, pt>2.0GeV, R = 0.3
    float ct3PtCut20() const {return ct3PtCut20_;}
    /// Track-based isolation, pt>2.0GeV, R = 0.4
    float ct4PtCut20() const {return ct4PtCut20_;}
    /// Track-based isolation, pt>2.0GeV, R = 0.5
    float ct5PtCut20() const {return ct5PtCut20_;}

    /// Swiss-Cross crystal ratio
    float swissCrx() const {return swissCrx_;}
    /// Ecal rechit seed time
    float seedTime() const {return seedTime_;}

    // setters

    /// Cluster-based isolation (ECAL) R = 0.1
    void cc1(float cc1)  {cc1_ = cc1;}
    /// Cluster-based isolation (ECAL) R = 0.2
    void cc2(float cc2)  {cc2_ = cc2;}
    /// Cluster-based isolation (ECAL) R = 0.3
    void cc3(float cc3)  {cc3_ = cc3;}
    /// Cluster-based isolation (ECAL) R = 0.4
    void cc4(float cc4)  {cc4_ = cc4;}
    /// Cluster-based isolation (ECAL) R = 0.5
    void cc5(float cc5)  {cc5_ = cc5;}

    /// Rechit-based isolation (HCAL) R = 0.1
    void cr1(float cr1)  {cr1_ = cr1;}
    /// Rechit-based isolation (HCAL) R = 0.2
    void cr2(float cr2)  {cr2_ = cr2;}
    /// Rechit-based isolation (HCAL) R = 0.3
    void cr3(float cr3)  {cr3_ = cr3;}
    /// Rechit-based isolation (HCAL) R = 0.4
    void cr4(float cr4)  {cr4_ = cr4;}
    /// Rechit-based isolation (HCAL) R = 0.5
    void cr5(float cr5)  {cr5_ = cr5;}

    /// Track-based isolation, pt>2.0GeV, R = 0.1
    void ct1PtCut20(float ct1PtCut20)  {ct1PtCut20_ = ct1PtCut20;}
    /// Track-based isolation, pt>2.0GeV, R = 0.2
    void ct2PtCut20(float ct2PtCut20)  {ct2PtCut20_ = ct2PtCut20;}
    /// Track-based isolation, pt>2.0GeV, R = 0.3
    void ct3PtCut20(float ct3PtCut20)  {ct3PtCut20_ = ct3PtCut20;}
    /// Track-based isolation, pt>2.0GeV, R = 0.4
    void ct4PtCut20(float ct4PtCut20)  {ct4PtCut20_ = ct4PtCut20;}
    /// Track-based isolation, pt>2.0GeV, R = 0.5
    void ct5PtCut20(float ct5PtCut20)  {ct5PtCut20_ = ct5PtCut20;}

    /// Swiss-Cross crystal ratio
    void swissCrx(float swissCrx)  {swissCrx_ = swissCrx;}
    /// Ecal rechit seed time
    void seedTime(float seedTime)  {seedTime_ = seedTime;}


  private:

    float cc1_, cc2_, cc3_, cc4_, cc5_;
    float cr1_, cr2_, cr3_, cr4_, cr5_;
    float ct1PtCut20_, ct2PtCut20_, ct3PtCut20_, ct4PtCut20_, ct5PtCut20_;

    float swissCrx_, seedTime_;
  };

  typedef edm::ValueMap<reco::HIPhotonIsolation> HIPhotonIsolationMap;

}
#endif
