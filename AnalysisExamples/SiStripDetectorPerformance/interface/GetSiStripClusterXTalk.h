// Declaration of function that calculates SiStripClusterXTalk
// [Note: code is taken from Tutorial:
//          [CMSSW]/UserCode/SamvelKhalatyan/Tutorial/Clusters
//        with minor changes]
//
// Author : Samvel Khalatyan (samvel at fnal dot gov)
// Created: 12/05/06
// Licence: GPL

#ifndef ANALYSISEXAMPLES_SISTRIPDETECTORPERFORMANCE_GET_SI_STRIP_CLUSTER_XTALK_H
#define ANALYSISEXAMPLES_SISTRIPDETECTORPERFORMANCE_GET_SI_STRIP_CLUSTER_XTALK_H

#include <vector>

#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/SiStripDigi/interface/SiStripDigi.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

namespace extra {
  typedef edm::DetSetVector<SiStripDigi> DSVSiStripDigis;
  
  //   Function that would calculate ClusterXTalk [see below for details].
  // ClusterXTalk will be calculated only in case ChargeL and ChargeR are
  // within N% error of its average value:
  //     
  //     abs( ChargeL - ChargeR) < 2 * N% * ( ChargeL + ChargeR) / 2
  //
  // [Note: use call of getSiStripClusterEta function with explicit use
  //        of GetClusteXTalk in case different XTALK_ERR should be used, e.g.:
  //          
  //          ... = getSiStripClusterEta( anAmplitudes,
  //                                      nFirstStrip,
  //                                      oDigis,
  //                                      GetClusterXTalk( 0.01));
  //
  //        where 0.01 stands for 1%]
  //
  // @argument
  //   1  Charge in prev strip
  //   2  Charge in current strip
  //   3  Charge in next strip
  // @return
  //   -10      if CHARGE_L and CHARGE_R are not within XTALK_ERR
  //   (0,1)  Calculated ClusterXTalk
  struct GetClusterXTalk {
    inline explicit GetClusterXTalk( const double &rdXTALK_ERR)
      : dXTALK_ERR( rdXTALK_ERR) {}
    
    inline double operator()( const uint16_t &rnCHARGE_L,
                              const uint16_t &rnCHARGE_M,
                              const uint16_t &rnCHARGE_R) const {

      return ( abs( rnCHARGE_L - rnCHARGE_R) < 
                 2 * dXTALK_ERR * ( rnCHARGE_L + rnCHARGE_R) / 2 &&
                 0 < rnCHARGE_M ? 
         1.0 * ( rnCHARGE_L + rnCHARGE_R) / rnCHARGE_M : 
         -10);
    }

    private:
      const double dXTALK_ERR;
  };

  //    Calculate SiStripClsuterXTalk:
  //        
  //         ClusterXTalk = ( ChargePrev + ChargeNext) / ChargeN
  //
  // very helpful value for understanding how signal affects neighbouring
  // strips. Given logic should be used in Monte-Carlo simulations.
  //
  // @argument
  //  1   SiStripCluster amplitudes collection
  //  2   First strip # in cluster, aka shift
  //  3   Collection of Digis
  //      [Note: make sure appropriate vector of DIGIS is passed. Suppose
  //             cluster belongs to DetId = N and you pass ALL Digis to the
  //             function. Then function will find the first Digi in vector
  //             with the same strip# as non-zero strip in OneStrip cluster.
  //             Found Digi may belong to any DetId that is != N (!!!). Take
  //             a look at helper function to get around mentioned odd]
  //    4   GetClusterXTalk with initial percent set
  // @return:
  //    -99    in case Strip Amplitudes vector is empty or does not hold any
  //           non zero umplitudes
  //    -10    if ChargeL and ChargeR are not within XTALK_ERR
  //    0      One non-zero strip cluster for which corresponding Digi was not
  //           found
  //    1      One non-zero strip cluster for which N-1 and N+1 Digi was not 
  //           found
  //    (0,1)  Calculated ClusterXTalk [Note: boundaries excluded]
  //
  //    [Note: look at the code for more details]
  double getSiStripClusterXTalk( const std::vector<uint16_t> 
                                   &roSTRIP_AMPLITUDES,
                                 const uint16_t                 &rnFIRST_STRIP,
                                 const std::vector<SiStripDigi> &roDIGIS,
                                 const GetClusterXTalk &roGetClusterXTalk = 
                                   GetClusterXTalk( 0.05));

  // Helper function
  inline double getSiStripClusterXTalk( const SiStripCluster &roCLUSTER,
                                        const 
                                          edm::Handle<DSVSiStripDigis> 
                                                        &roDSVDigis,
                                        const GetClusterXTalk 
                                          &roGetClusterXTalk = 
                                            GetClusterXTalk( 0.05)) {

    return getSiStripClusterXTalk( roCLUSTER.amplitudes(),
                                   roCLUSTER.firstStrip(),
           roDSVDigis->operator[]( 
             roCLUSTER.geographicalId()).data);
  }
} // End namespace extra

#endif 
  // ANALYSISEXAMPLES_SISTRIPDETECTORPERFORMANCE_GET_SI_STRIP_CLUSTER_XTALK_H
