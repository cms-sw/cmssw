#ifndef __l1microgmtlutfactories_h
#define __l1microgmtlutfactories_h

#include <iostream>

#include "L1Trigger/L1TMuon/interface/MicroGMTRankPtQualLUT.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTMatchQualLUT.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTExtrapolationLUT.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTCaloIndexSelectionLUT.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTAbsoluteIsolationCheckLUT.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTRelativeIsolationCheckLUT.h"
#include "CondFormats/L1TObjects/interface/L1TMuonGlobalParams.h"

namespace l1t {
  class MicroGMTRankPtQualLUTFactory {
    public:
      MicroGMTRankPtQualLUTFactory() {};
      ~MicroGMTRankPtQualLUTFactory() {};

      typedef std::shared_ptr<MicroGMTRankPtQualLUT> ReturnType;

      static ReturnType create(const std::string& filename, const int fwVersion, const unsigned ptFactor, const unsigned qualFactor);
      static ReturnType create(l1t::LUT* lut, const int fwVersion);
  };

  class MicroGMTMatchQualLUTFactory {
    public:
      MicroGMTMatchQualLUTFactory() {};
      ~MicroGMTMatchQualLUTFactory() {};

      typedef std::shared_ptr<MicroGMTMatchQualLUT> ReturnType;

      static ReturnType create(const std::string& filename, const double maxDR, const double fEta, const double fEtaCoarse, const double fPhi, cancel_t cancelType, const int fwVersion);
      static ReturnType create(l1t::LUT* lut, cancel_t cancelType, const int fwVersion);
  };

  class MicroGMTExtrapolationLUTFactory {
    public:
      MicroGMTExtrapolationLUTFactory() {};
      ~MicroGMTExtrapolationLUTFactory() {};

      typedef std::shared_ptr<MicroGMTExtrapolationLUT> ReturnType;

      static ReturnType create(const std::string& filename, const int type, const int fwVersion);
      static ReturnType create(l1t::LUT* lut, const int type, const int fwVersion);
  };

  class MicroGMTCaloIndexSelectionLUTFactory {
    public:
      MicroGMTCaloIndexSelectionLUTFactory() {};
      ~MicroGMTCaloIndexSelectionLUTFactory() {};

      typedef std::shared_ptr<MicroGMTCaloIndexSelectionLUT> ReturnType;

      static ReturnType create(const std::string& filename, const int type, const int fwVersion);
      static ReturnType create(l1t::LUT* lut, const int type, const int fwVersion);
  };

  class MicroGMTAbsoluteIsolationCheckLUTFactory {
    public:
      MicroGMTAbsoluteIsolationCheckLUTFactory() {};
      ~MicroGMTAbsoluteIsolationCheckLUTFactory() {};

      typedef std::shared_ptr<MicroGMTAbsoluteIsolationCheckLUT> ReturnType;

      static ReturnType create(const std::string& filename, const int fwVersion);
      static ReturnType create(l1t::LUT* lut, const int fwVersion);
  };

  class MicroGMTRelativeIsolationCheckLUTFactory {
    public:
      MicroGMTRelativeIsolationCheckLUTFactory() {};
      ~MicroGMTRelativeIsolationCheckLUTFactory() {};

      typedef std::shared_ptr<MicroGMTRelativeIsolationCheckLUT> ReturnType;

      static ReturnType create(const std::string& filename, const int fwVersion);
      static ReturnType create(l1t::LUT* lut, const int fwVersion);
  };
}

#endif /* defined(__l1microgmtlutfactories_h) */
