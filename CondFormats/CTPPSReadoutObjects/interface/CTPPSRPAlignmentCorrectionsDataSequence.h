/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors:
 *	Jan Kašpar (jan.kaspar@gmail.com)
 *
 ****************************************************************************/

#ifndef CondFormats_CTPPSReadoutObjects_CTPPSRPAlignmentCorrectionsDataSequence
#define CondFormats_CTPPSReadoutObjects_CTPPSRPAlignmentCorrectionsDataSequence

#include <vector>

#include "FWCore/Framework/interface/ValidityInterval.h"

#include "CondFormats/CTPPSReadoutObjects/interface/CTPPSRPAlignmentCorrectionsData.h"

/**
 *\brief Time sequence of alignment corrections.
 * I/O methods have been factored out to:
 *   CondFormats/CTPPSReadoutObjects/interface/CTPPSRPAlignmentCorrectionsMethods.h
 */
class CTPPSRPAlignmentCorrectionsDataSequence
    : public std::vector<std::pair<edm::ValidityInterval, CTPPSRPAlignmentCorrectionsData> > {
public:
  CTPPSRPAlignmentCorrectionsDataSequence() {}

  void insert(const edm::ValidityInterval &iov, const CTPPSRPAlignmentCorrectionsData &data) {
    emplace_back(iov, data);
  }
};

#endif
