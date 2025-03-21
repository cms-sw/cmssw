
#include "FWCore/ParameterSet/interface/EmptyGroupDescription.h"
#include "FWCore/ParameterSet/interface/DocFormatHelper.h"

#include <ostream>

namespace edm {

  EmptyGroupDescription::EmptyGroupDescription() {}

  void EmptyGroupDescription::checkAndGetLabelsAndTypes_(std::set<std::string>& /*usedLabels*/,
                                                         std::set<ParameterTypes>& /*parameterTypes*/,
                                                         std::set<ParameterTypes>& /*wildcardTypes*/) const {}

  void EmptyGroupDescription::validate_(ParameterSet&,
                                        std::set<std::string>& /*validatedLabels*/,
                                        Modifier /*optional*/) const {}

  void EmptyGroupDescription::writeCfi_(std::ostream&,
                                        Modifier /*optional*/,
                                        bool& /*startWithComma*/,
                                        int /*indentation*/,
                                        CfiOptions&,
                                        bool& /*wroteSomething*/) const {}

  void EmptyGroupDescription::print_(std::ostream& os,
                                     Modifier /*optional*/,
                                     bool /*writeToCfi*/,
                                     DocFormatHelper& dfh) const {
    if (dfh.pass() == 1) {
      dfh.indent(os);
      os << "Empty group description\n";

      if (!dfh.brief()) {
        os << "\n";
      }
    }
  }

  bool EmptyGroupDescription::exists_(ParameterSet const&) const { return true; }

  bool EmptyGroupDescription::partiallyExists_(ParameterSet const& pset) const { return exists(pset); }

  int EmptyGroupDescription::howManyXORSubNodesExist_(ParameterSet const& pset) const { return exists(pset) ? 1 : 0; }
}  // namespace edm
