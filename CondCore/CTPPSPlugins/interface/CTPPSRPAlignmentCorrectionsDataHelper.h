/****************************************************************************
 *
 * This is a part of PPS PI software.
 *
 ****************************************************************************/

#ifndef CONDCORE_CTPPSPLUGINS_CTPPSRPALIGNMENTCORRECTIONSDATAHELPER_H
#define CONDCORE_CTPPSPLUGINS_CTPPSRPALIGNMENTCORRECTIONSDATAHELPER_H

// User includes
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CondCore/Utilities/interface/PayloadInspectorModule.h"
#include "CondCore/Utilities/interface/PayloadInspector.h"
#include "CondCore/CondDB/interface/Time.h"
#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsData.h"
#include "DataFormats/CTPPSDetId/interface/CTPPSDetId.h"

// system includes
#include <memory>
#include <sstream>

class CTPPSRPAlignment {
public:
  enum RP { RP3 = 3, RP23 = 23, RP103 = 103, RP123 = 123 };

  enum Shift { x = 1, y = 2 };

  static std::string getStringFromRPEnum(const RP& rp) {
    switch (rp) {
      case 3:
        return "RP 3";
      case 23:
        return "RP 23";
      case 103:
        return "RP 103";
      case 123:
        return "RP 123";

      default:
        return "not here";
    }
  }

  static std::string getStringFromShiftEnum(const Shift& sh, bool unc) {
    switch (sh) {
      case 1:
        if (unc)
          return "x shift uncertainty";
        else
          return "x shift";
      case 2:
        if (unc)
          return "y shift uncertainty";
        else
          return "y shift";

      default:
        return "not here";
    }
  }
};

/************************************************
    History plots
*************************************************/
template <CTPPSRPAlignment::RP rp, CTPPSRPAlignment::Shift sh, bool unc, class PayloadType>
class RPShift_History : public cond::payloadInspector::HistoryPlot<PayloadType, float> {
public:
  RPShift_History()
      : cond::payloadInspector::HistoryPlot<PayloadType, float>(
            CTPPSRPAlignment::getStringFromRPEnum(rp) + " " + CTPPSRPAlignment::getStringFromShiftEnum(sh, unc) +
                " [mm] vs. Runs",
            CTPPSRPAlignment::getStringFromRPEnum(rp) + " " + CTPPSRPAlignment::getStringFromShiftEnum(sh, unc) +
                " [mm]") {}

  uint decodeRP(uint r) {
    if (r == CTPPSRPAlignment::RP3 || r == CTPPSRPAlignment::RP23 || r == CTPPSRPAlignment::RP103 ||
        r == CTPPSRPAlignment::RP123)
      return r;
    else {
      CTPPSDetId* config = new CTPPSDetId(r);
      int potId = config->arm() * 100 + config->station() * 10 + config->rp();
      delete config;
      return potId;
    }
  }

  //uncertainty graphs should be considered as +\- getShXUnc() uncertainty value
  //in case record does not exist it returns nonsense -1 value
  float getFromPayload(PayloadType& payload) override {
    for (auto& configuration : payload.getRPMap()) {
      if (decodeRP(configuration.first) == rp) {
        if (unc) {
          if (sh == 1)
            return configuration.second.getShXUnc();
          if (sh == 2)
            return configuration.second.getShYUnc();
        } else {
          if (sh == 1)
            return configuration.second.getShX();
          if (sh == 2)
            return configuration.second.getShY();
        }
      }
    }
    if (unc)
      return -1;
    return 0;
  }
};

#endif
