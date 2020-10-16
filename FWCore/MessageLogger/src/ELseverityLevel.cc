// ----------------------------------------------------------------------
//
// ELseverityLevel.cc - implement objects that encode a message's urgency
//
//      Both frameworker and user will often pass one of the
//      instantiated severity levels to logger methods.
//
//      The only other methods of ELseverityLevel a frameworker
//      might use is to check the relative level of two severities
//      using operator<() or the like.
//
// ----------------------------------------------------------------------

#include <cassert>
#include <ostream>

#include "FWCore/MessageLogger/interface/ELseverityLevel.h"
#include "FWCore/MessageLogger/interface/ELmap.h"

namespace {

  // ----------------------------------------------------------------------
  // Helper to construct the string->ELsev_ map on demand:
  // ----------------------------------------------------------------------

  typedef std::map<std::string const, edm::ELseverityLevel::ELsev_, std::less<>> ELmap;

  ELmap const& loadMap() {
    static const ELmap m = {{edm::ELzeroSeverity.getSymbol(), edm::ELseverityLevel::ELsev_zeroSeverity},
                            {edm::ELzeroSeverity.getName(), edm::ELseverityLevel::ELsev_zeroSeverity},
                            {edm::ELzeroSeverity.getInputStr(), edm::ELseverityLevel::ELsev_zeroSeverity},
                            {edm::ELzeroSeverity.getVarName(), edm::ELseverityLevel::ELsev_zeroSeverity},
                            {edm::ELdebug.getSymbol(), edm::ELseverityLevel::ELsev_success},
                            {edm::ELdebug.getName(), edm::ELseverityLevel::ELsev_success},
                            {edm::ELdebug.getInputStr(), edm::ELseverityLevel::ELsev_success},
                            {edm::ELdebug.getVarName(), edm::ELseverityLevel::ELsev_success},
                            {edm::ELinfo.getSymbol(), edm::ELseverityLevel::ELsev_info},
                            {edm::ELinfo.getName(), edm::ELseverityLevel::ELsev_info},
                            {edm::ELinfo.getInputStr(), edm::ELseverityLevel::ELsev_info},
                            {edm::ELinfo.getVarName(), edm::ELseverityLevel::ELsev_info},
                            {edm::ELfwkInfo.getSymbol(), edm::ELseverityLevel::ELsev_fwkInfo},
                            {edm::ELfwkInfo.getName(), edm::ELseverityLevel::ELsev_fwkInfo},
                            {edm::ELfwkInfo.getInputStr(), edm::ELseverityLevel::ELsev_fwkInfo},
                            {edm::ELfwkInfo.getVarName(), edm::ELseverityLevel::ELsev_fwkInfo},
                            {edm::ELwarning.getSymbol(), edm::ELseverityLevel::ELsev_warning},
                            {edm::ELwarning.getName(), edm::ELseverityLevel::ELsev_warning},
                            {edm::ELwarning.getInputStr(), edm::ELseverityLevel::ELsev_warning},
                            {edm::ELwarning.getVarName(), edm::ELseverityLevel::ELsev_warning},
                            {edm::ELerror.getSymbol(), edm::ELseverityLevel::ELsev_error},
                            {edm::ELerror.getName(), edm::ELseverityLevel::ELsev_error},
                            {edm::ELerror.getInputStr(), edm::ELseverityLevel::ELsev_error},
                            {edm::ELerror.getVarName(), edm::ELseverityLevel::ELsev_error},
                            {edm::ELunspecified.getSymbol(), edm::ELseverityLevel::ELsev_unspecified},
                            {edm::ELunspecified.getName(), edm::ELseverityLevel::ELsev_unspecified},
                            {edm::ELunspecified.getInputStr(), edm::ELseverityLevel::ELsev_unspecified},
                            {edm::ELunspecified.getVarName(), edm::ELseverityLevel::ELsev_unspecified},
                            {edm::ELsevere.getSymbol(), edm::ELseverityLevel::ELsev_severe},
                            {edm::ELsevere.getName(), edm::ELseverityLevel::ELsev_severe},
                            {edm::ELsevere.getInputStr(), edm::ELseverityLevel::ELsev_severe},
                            {edm::ELsevere.getVarName(), edm::ELseverityLevel::ELsev_severe},
                            {edm::ELhighestSeverity.getSymbol(), edm::ELseverityLevel::ELsev_highestSeverity},
                            {edm::ELhighestSeverity.getName(), edm::ELseverityLevel::ELsev_highestSeverity},
                            {edm::ELhighestSeverity.getInputStr(), edm::ELseverityLevel::ELsev_highestSeverity},
                            {edm::ELhighestSeverity.getVarName(), edm::ELseverityLevel::ELsev_highestSeverity}};

    return m;
  }
}  // namespace

namespace edm {
  // ----------------------------------------------------------------------
  // Birth/death:
  // ----------------------------------------------------------------------

  ELseverityLevel::ELseverityLevel(std::string_view s) {
    static ELmap const& m = loadMap();

    ELmap::const_iterator i = m.find(s);
    myLevel = (i == m.end()) ? ELsev_unspecified : i->second;
  }

  // ----------------------------------------------------------------------
  // Accessors:
  // ----------------------------------------------------------------------

  const std::string& ELseverityLevel::getSymbol() const {
    static const auto symbols = []() {
      std::array<std::string, nLevels> ret;
      ret[ELsev_noValueAssigned] = "0";
      ret[ELsev_zeroSeverity] = "--";
      ret[ELsev_success] = "-d";
      ret[ELsev_info] = "-i";
      ret[ELsev_fwkInfo] = "-f";
      ret[ELsev_warning] = "-w";
      ret[ELsev_error] = "-e";
      ret[ELsev_unspecified] = "??";
      ret[ELsev_severe] = "-s";
      ret[ELsev_highestSeverity] = "!!";
      return ret;
    }();

    assert(myLevel < nLevels);
    return symbols[myLevel];
  }

  const std::string& ELseverityLevel::getName() const {
    static const auto names = []() {
      std::array<std::string, nLevels> ret;
      ret[ELsev_noValueAssigned] = "?no value?";
      ret[ELsev_zeroSeverity] = "--";
      ret[ELsev_success] = "Debug";
      ret[ELsev_info] = "Info";
      ret[ELsev_fwkInfo] = "FwkInfo";
      ret[ELsev_warning] = "Warning";
      ret[ELsev_error] = "Error";
      ret[ELsev_unspecified] = "??";
      ret[ELsev_severe] = "System";
      ret[ELsev_highestSeverity] = "!!";
      return ret;
    }();

    assert(myLevel < nLevels);
    return names[myLevel];
  }

  const std::string& ELseverityLevel::getInputStr() const {
    static const auto inputs = []() {
      std::array<std::string, nLevels> ret;
      ret[ELsev_noValueAssigned] = "?no value?";
      ret[ELsev_zeroSeverity] = "ZERO";
      ret[ELsev_success] = "DEBUG";
      ret[ELsev_info] = "INFO";
      ret[ELsev_fwkInfo] = "FWKINFO";
      ret[ELsev_warning] = "WARNING";
      ret[ELsev_error] = "ERROR";
      ret[ELsev_unspecified] = "UNSPECIFIED";
      ret[ELsev_severe] = "SYSTEM";
      ret[ELsev_highestSeverity] = "HIGHEST";
      return ret;
    }();

    assert(myLevel < nLevels);
    return inputs[myLevel];
  }

  const std::string& ELseverityLevel::getVarName() const {
    static const auto varNames = []() {
      std::array<std::string, nLevels> ret;
      ret[ELsev_noValueAssigned] = "?no value?";
      ret[ELsev_zeroSeverity] = "ELzeroSeverity   ";
      ret[ELsev_success] = "ELdebug          ";
      ret[ELsev_info] = "ELinfo           ";
      ret[ELsev_fwkInfo] = "ELfwkInfo        ";
      ret[ELsev_warning] = "ELwarning        ";
      ret[ELsev_error] = "ELerror          ";
      ret[ELsev_unspecified] = "ELunspecified    ";
      ret[ELsev_severe] = "ELsystem         ";
      ret[ELsev_highestSeverity] = "ELhighestSeverity";
      return ret;
    }();

    assert(myLevel < nLevels);
    return varNames[myLevel];
  }

  // ----------------------------------------------------------------------
  // Emitter:
  // ----------------------------------------------------------------------

  std::ostream& operator<<(std::ostream& os, const ELseverityLevel& sev) { return os << " -" << sev.getName() << "- "; }

}  // end of namespace edm  */
