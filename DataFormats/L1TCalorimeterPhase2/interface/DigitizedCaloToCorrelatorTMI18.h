#ifndef DataFormats_L1TCalorimeterPhase2_DigitizedCaloToCorrelatorTMI18_h
#define DataFormats_L1TCalorimeterPhase2_DigitizedCaloToCorrelatorTMI18_h

#include <ap_int.h>
#include <vector>
#include "DataFormats/L1TCalorimeterPhase2/interface/GCTEmDigiCluster.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/GCTHadDigiCluster.h"
#include "DataFormats/L1TCalorimeterPhase2/interface/DigitizedClusterCorrelator.h"

namespace l1tp2 {

  typedef std::variant<std::monostate, l1tp2::GCTEmDigiCluster, l1tp2::GCTHadDigiCluster> GCTDigiCluster;
  typedef std::vector<l1tp2::GCTDigiCluster> GCTDigiClusterLink;

  class DigitizedCaloToCorrelatorTMI18 {
  private:
    // Data
    ap_uint<64> Card0Data[162] ;
    ap_uint<64> Card1Data[162] ;
    ap_uint<64> Card2Data[162] ;

    GCTDigiClusterLink Card0Link;
    GCTDigiClusterLink Card1Link;
    GCTDigiClusterLink Card2Link;

  public:

    DigitizedCaloToCorrelatorTMI18() { 
	    for (int i = 0; i < 162; i++) {
		    Card0Data[i]=0;
		    Card1Data[i]=0;
		    Card2Data[i]=0;
	    }
    }
    DigitizedCaloToCorrelatorTMI18(ap_uint<64> data0[162], ap_uint<64> data1[162], ap_uint<64> data2[162], GCTDigiClusterLink link0, GCTDigiClusterLink link1, GCTDigiClusterLink link2) { 
	    for (int i = 0; i < 162; i++) {
		    Card0Data[i]=data0[i];
		    Card1Data[i]=data1[i];
		    Card2Data[i]=data2[i];
	    }
	    Card0Link = link0;
	    Card1Link = link1;
	    Card2Link = link2;
    }

    const ap_uint<64>*  dataCard0() const { return Card0Data; }
    const ap_uint<64>*  dataCard1() const { return Card1Data; }
    const ap_uint<64>*  dataCard2() const { return Card2Data; }
    const GCTDigiClusterLink& linkCard0() const { return Card0Link; }
    const GCTDigiClusterLink& linkCard1() const { return Card1Link; }
    const GCTDigiClusterLink& linkCard2() const { return Card2Link; } 

  };

  // Collection typedef
  // this represents both the EM and PF clusters from a single GCT card in one link
  typedef std::vector<l1tp2::DigitizedCaloToCorrelatorTMI18> DigitizedCaloToCorrelatorCollectionTMI18;

}  // namespace l1tp2

#endif
