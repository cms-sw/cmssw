/****************************************************************************
 * Authors:
 *  Jan Kaspar (jan.kaspar@gmail.com)
 *  Helena Malbouisson
 *  Clemencia Mora Herrera
 *  Christopher Misan
 ****************************************************************************/

namespace edm {
  class ParameterSet;
}

#include "CondFormats/PPSObjects/interface/CTPPSRPAlignmentCorrectionsDataSequence.h"

#include <vector>
#include <string>

class CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon {
public:
  CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon(const edm::ParameterSet &p);
  ~CTPPSRPAlignmentCorrectionsDataESSourceXMLCommon();

  CTPPSRPAlignmentCorrectionsDataSequence acsMeasured, acsReal, acsMisaligned;
  CTPPSRPAlignmentCorrectionsData acMeasured, acReal, acMisaligned;

  unsigned int verbosity;

  static edm::EventID previousLS(const edm::EventID &src);
  static edm::EventID nextLS(const edm::EventID &src);

protected:
  CTPPSRPAlignmentCorrectionsDataSequence Merge(const std::vector<CTPPSRPAlignmentCorrectionsDataSequence> &) const;

  void PrepareSequence(const std::string &label,
                       CTPPSRPAlignmentCorrectionsDataSequence &seq,
                       const std::vector<std::string> &files) const;
};