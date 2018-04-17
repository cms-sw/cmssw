/****************************************************************************
 *
 * This is a part of TOTEM offline software.
 * Authors:
 *	Jan Kašpar (jan.kaspar@gmail.com)
 *
 ****************************************************************************/

#ifndef DataFormats_CTPPSAlignment_RPAlignmentCorrectionsDataSequence
#define DataFormats_CTPPSAlignment_RPAlignmentCorrectionsDataSequence

#include <vector>

#include "FWCore/Framework/interface/ValidityInterval.h"

#include "DataFormats/CTPPSAlignment/interface/RPAlignmentCorrectionsData.h"

/**
 *\brief Time sequence of alignment corrections.
 * I/O methods have been factored out to:
 *   Geometry/VeryForwardGeometryBuilder/interface/RPAlignmentCorrectionsMethods.h
 */
class RPAlignmentCorrectionsDataSequence : public std::vector< std::pair<edm::ValidityInterval, RPAlignmentCorrectionsData> >
{
  public:
    RPAlignmentCorrectionsDataSequence() {}

    void insert(const edm::ValidityInterval &iov, const RPAlignmentCorrectionsData &data)
    {
      emplace_back(iov, data);
    }
};

#endif
