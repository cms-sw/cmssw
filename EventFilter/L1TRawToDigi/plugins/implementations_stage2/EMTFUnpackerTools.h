// Tools for unpacking and packing EMTF data

#ifndef EventFilter_L1TRawToDigi_EMTFUnpackerTools_h
#define EventFilter_L1TRawToDigi_EMTFUnpackerTools_h

// Generally useful includes
#include <iostream>
#include <iomanip>  // For things like std::setw
#include <array>
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/L1TMuon/interface/EMTFDaqOut.h"
#include "DataFormats/L1TMuon/interface/EMTFHit.h"
#include "DataFormats/L1TMuon/interface/EMTFTrack.h"
#include "L1Trigger/L1TMuonEndCap/interface/TrackTools.h"

namespace l1t {
  namespace stage2 {
    namespace emtf {
      namespace L1TMuonEndCap =
          ::emtf;  // use alias 'L1TMuonEndCap' for the namespace 'emtf' used in L1Trigger/L1TMuonEndCap

      void ImportME(EMTFHit& _hit, const l1t::emtf::ME _ME, const int _endcap, const int _evt_sector);
      void ImportRPC(EMTFHit& _hit, const l1t::emtf::RPC _RPC, const int _endcap, const int _evt_sector);
      void ImportGEM(EMTFHit& _hit, const l1t::emtf::GEM& _GEM, const int _endcap, const int _evt_sector);
      void ImportSP(EMTFTrack& _track, const l1t::emtf::SP _SP, const int _endcap, const int _evt_sector);

      // Integer version of pow() - returns base^exp
      inline int PowInt(int base, int exp) {
        if (exp == 0)
          return 1;
        if (exp == 1)
          return base;
        return base * PowInt(base, exp - 1);
      }

      // Compute the two's complement of an integer
      inline int TwosCompl(int nBits, int bits) {
        if (bits >> (nBits - 1) == 0)
          return bits;
        else
          return bits - PowInt(2, nBits);
      };

      // Get the integer value of specified bits from a 16-bit word (0xffff)
      inline uint16_t GetHexBits(uint16_t word, uint16_t lowBit, uint16_t highBit) {
        return ((word >> lowBit) & (PowInt(2, (1 + highBit - lowBit)) - 1));
      }

      // Get the integer value of specified bits from a 32-bit word (0xffffffff)
      inline uint32_t GetHexBits(uint32_t word, uint32_t lowBit, uint32_t highBit) {
        return ((word >> lowBit) & (PowInt(2, (1 + highBit - lowBit)) - 1));
      }

      // Get the integer value of specified bits from two 16-bit words (0xffff, 0xffff)
      inline uint32_t GetHexBits(
          uint16_t word1, uint16_t lowBit1, uint16_t highBit1, uint16_t word2, uint16_t lowBit2, uint16_t highBit2) {
        uint16_t word1_sel = (word1 >> lowBit1) & (PowInt(2, (1 + highBit1 - lowBit1)) - 1);
        uint16_t word2_sel = (word2 >> lowBit2) & (PowInt(2, (1 + highBit2 - lowBit2)) - 1);
        return ((word2_sel << (1 + highBit1 - lowBit1)) | word1_sel);
      }

    }  // End namespace emtf
  }    // End namespace stage2
}  // End namespace l1t

#endif /* define EventFilter_L1TRawToDigi_EMTFUnpackerTools_h */
