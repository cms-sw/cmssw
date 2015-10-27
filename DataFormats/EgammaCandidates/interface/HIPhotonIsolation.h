#ifndef EgammaCandidates_HIPhotonIsolation_h
#define EgammaCandidates_HIPhotonIsolation_h

#include "DataFormats/Common/interface/ValueMap.h"

namespace reco{

  class HIPhotonIsolation {

  public:

  HIPhotonIsolation() : ecalClusterIsoR1_(0),
      ecalClusterIsoR2_(0),
      ecalClusterIsoR3_(0),
      ecalClusterIsoR4_(0),
      ecalClusterIsoR5_(0),
      hcalRechitIsoR1_(0),
      hcalRechitIsoR2_(0),
      hcalRechitIsoR3_(0),
      hcalRechitIsoR4_(0),
      hcalRechitIsoR5_(0),
      trackIsoR1PtCut20_(0),
      trackIsoR2PtCut20_(0),
      trackIsoR3PtCut20_(0),
      trackIsoR4PtCut20_(0),
      trackIsoR5PtCut20_(0),
      swissCrx_(0),
      seedTime_(0)
      {}
    virtual ~HIPhotonIsolation() {}

    //getters

    /// Cluster-based isolation (ECAL) R = 0.1
    float ecalClusterIsoR1() const {return ecalClusterIsoR1_;}
    /// Cluster-based isolation (ECAL) R = 0.2
    float ecalClusterIsoR2() const {return ecalClusterIsoR2_;}
    /// Cluster-based isolation (ECAL) R = 0.3
    float ecalClusterIsoR3() const {return ecalClusterIsoR3_;}
    /// Cluster-based isolation (ECAL) R = 0.4
    float ecalClusterIsoR4() const {return ecalClusterIsoR4_;}
    /// Cluster-based isolation (ECAL) R = 0.5
    float ecalClusterIsoR5() const {return ecalClusterIsoR5_;}

    /// Rechit-based isolation (HCAL) R = 0.1
    float hcalRechitIsoR1() const {return hcalRechitIsoR1_;}
    /// Rechit-based isolation (HCAL) R = 0.2
    float hcalRechitIsoR2() const {return hcalRechitIsoR2_;}
    /// Rechit-based isolation (HCAL) R = 0.3
    float hcalRechitIsoR3() const {return hcalRechitIsoR3_;}
    /// Rechit-based isolation (HCAL) R = 0.4
    float hcalRechitIsoR4() const {return hcalRechitIsoR4_;}
    /// Rechit-based isolation (HCAL) R = 0.5
    float hcalRechitIsoR5() const {return hcalRechitIsoR5_;}

    /// Track-based isolation, pt>2.0GeV, R = 0.1
    float trackIsoR1PtCut20() const {return trackIsoR1PtCut20_;}
    /// Track-based isolation, pt>2.0GeV, R = 0.2
    float trackIsoR2PtCut20() const {return trackIsoR2PtCut20_;}
    /// Track-based isolation, pt>2.0GeV, R = 0.3
    float trackIsoR3PtCut20() const {return trackIsoR3PtCut20_;}
    /// Track-based isolation, pt>2.0GeV, R = 0.4
    float trackIsoR4PtCut20() const {return trackIsoR4PtCut20_;}
    /// Track-based isolation, pt>2.0GeV, R = 0.5
    float trackIsoR5PtCut20() const {return trackIsoR5PtCut20_;}

    /// SwissCross crystal ratio
    float swissCrx() const {return swissCrx_;}
    /// Ecal rechit seed time
    float seedTime() const {return seedTime_;}

    // setters

    /// Cluster-based isolation (ECAL) R = 0.1
    void ecalClusterIsoR1(float ecalClusterIsoR1)  {ecalClusterIsoR1_ = ecalClusterIsoR1;}
    /// Cluster-based isolation (ECAL) R = 0.2
    void ecalClusterIsoR2(float ecalClusterIsoR2)  {ecalClusterIsoR2_ = ecalClusterIsoR2;}
    /// Cluster-based isolation (ECAL) R = 0.3
    void ecalClusterIsoR3(float ecalClusterIsoR3)  {ecalClusterIsoR3_ = ecalClusterIsoR3;}
    /// Cluster-based isolation (ECAL) R = 0.4
    void ecalClusterIsoR4(float ecalClusterIsoR4)  {ecalClusterIsoR4_ = ecalClusterIsoR4;}
    /// Cluster-based isolation (ECAL) R = 0.5
    void ecalClusterIsoR5(float ecalClusterIsoR5)  {ecalClusterIsoR5_ = ecalClusterIsoR5;}

    /// Rechit-based isolation (HCAL) R = 0.1
    void hcalRechitIsoR1(float hcalRechitIsoR1)  {hcalRechitIsoR1_ = hcalRechitIsoR1;}
    /// Rechit-based isolation (HCAL) R = 0.2
    void hcalRechitIsoR2(float hcalRechitIsoR2)  {hcalRechitIsoR2_ = hcalRechitIsoR2;}
    /// Rechit-based isolation (HCAL) R = 0.3
    void hcalRechitIsoR3(float hcalRechitIsoR3)  {hcalRechitIsoR3_ = hcalRechitIsoR3;}
    /// Rechit-based isolation (HCAL) R = 0.4
    void hcalRechitIsoR4(float hcalRechitIsoR4)  {hcalRechitIsoR4_ = hcalRechitIsoR4;}
    /// Rechit-based isolation (HCAL) R = 0.5
    void hcalRechitIsoR5(float hcalRechitIsoR5)  {hcalRechitIsoR5_ = hcalRechitIsoR5;}

    /// Track-based isolation, pt>2.0GeV, R = 0.1
    void trackIsoR1PtCut20(float trackIsoR1PtCut20)  {trackIsoR1PtCut20_ = trackIsoR1PtCut20;}
    /// Track-based isolation, pt>2.0GeV, R = 0.2
    void trackIsoR2PtCut20(float trackIsoR2PtCut20)  {trackIsoR2PtCut20_ = trackIsoR2PtCut20;}
    /// Track-based isolation, pt>2.0GeV, R = 0.3
    void trackIsoR3PtCut20(float trackIsoR3PtCut20)  {trackIsoR3PtCut20_ = trackIsoR3PtCut20;}
    /// Track-based isolation, pt>2.0GeV, R = 0.4
    void trackIsoR4PtCut20(float trackIsoR4PtCut20)  {trackIsoR4PtCut20_ = trackIsoR4PtCut20;}
    /// Track-based isolation, pt>2.0GeV, R = 0.5
    void trackIsoR5PtCut20(float trackIsoR5PtCut20)  {trackIsoR5PtCut20_ = trackIsoR5PtCut20;}

    /// SwissCross ecal crystal ratio
    void swissCrx(float swissCrx)  {swissCrx_ = swissCrx;}
    /// Ecal rechit seed time
    void seedTime(float seedTime)  {seedTime_ = seedTime;}


  private:

    float ecalClusterIsoR1_, ecalClusterIsoR2_, ecalClusterIsoR3_, ecalClusterIsoR4_, ecalClusterIsoR5_;
    float hcalRechitIsoR1_, hcalRechitIsoR2_, hcalRechitIsoR3_, hcalRechitIsoR4_, hcalRechitIsoR5_;
    float trackIsoR1PtCut20_, trackIsoR2PtCut20_, trackIsoR3PtCut20_, trackIsoR4PtCut20_, trackIsoR5PtCut20_;

    float swissCrx_, seedTime_;
  };

  typedef edm::ValueMap<reco::HIPhotonIsolation> HIPhotonIsolationMap;

}
#endif
