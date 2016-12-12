#ifndef __l1microgmtconfiguration_h
#define __l1microgmtconfiguration_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"
#include "DataFormats/L1TMuon/interface/MuonCaloSumFwd.h"
#include "L1Trigger/L1TMuon/interface/GMTInternalMuonFwd.h"

#include <map>
#include <utility>

namespace l1t {
  class MicroGMTConfiguration {
    public:
      // All possible inputs for LUTs
      enum input_t {
        PT, PT_COARSE, PHI, ETA, ETA_COARSE, QUALITY, DELTA_ETA_RED, DELTA_PHI_RED, ENERGYSUM, ETA_FINE_BIT
      };
      enum output_t {
        ETA_OUT, PHI_OUT
      };

      typedef std::pair<input_t, int> PortType;
      typedef RegionalMuonCandBxCollection InputCollection;
      typedef MuonBxCollection OutputCollection;
      typedef Muon OutMuon;
      typedef GMTInternalMuon InterMuon;
      typedef GMTInternalMuonCollection InterMuonCollection;
      typedef GMTInternalMuonList InterMuonList;
      typedef MuonCaloSum CaloInput;
      typedef MuonCaloSumBxCollection CaloInputCollection;
      // Two's complement for a given bit-length
      static unsigned getTwosComp(const int signedInt, const int width);

      static int calcGlobalPhi(int locPhi, tftype t, int proc);

      static int setOutputMuonQuality(int muQual, tftype type, int haloBit);
  };
}
#endif /* defined (__l1microgmtconfiguration_h) */
