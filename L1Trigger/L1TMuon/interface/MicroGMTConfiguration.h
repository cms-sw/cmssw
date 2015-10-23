#ifndef __l1microgmtconfiguration_h
#define __l1microgmtconfiguration_h

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/GMTInternalMuon.h"
#include "DataFormats/L1TMuon/interface/GMTInputCaloSum.h"

#include <map>
#include <utility>

namespace l1t {
  class MicroGMTConfiguration {
    public:
      // All possible inputs for LUTs
      enum input_t {
        PT, PT_COARSE, PHI, ETA, ETA_COARSE, QUALITY, DELTA_ETA_RED, DELTA_PHI_RED
      };

      typedef std::pair<input_t, int> PortType;
      typedef RegionalMuonCandBxCollection InputCollection;
      typedef MuonBxCollection OutputCollection;
      typedef Muon OutMuon;
      typedef GMTInternalMuon InterMuon;
      typedef GMTInternalMuonCollection InterMuonCollection;
      typedef GMTInternalMuonList InterMuonList;
      typedef GMTInputCaloSum CaloInput;
      typedef GMTInputCaloSumBxCollection CaloInputCollection;
      // Two's complement for a given bit-length
      static unsigned getTwosComp(const int signedInt, const int width);

      static int calcGlobalPhi(int locPhi, tftype t, int proc);
  };
}
#endif /* defined (__l1microgmtconfiguration_h) */
