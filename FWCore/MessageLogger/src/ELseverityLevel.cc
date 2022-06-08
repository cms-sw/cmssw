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

#include <array>
#include <cassert>
#include <ostream>

#include "FWCore/MessageLogger/interface/ELseverityLevel.h"
#include "FWCore/MessageLogger/interface/ELmap.h"

namespace {
  using namespace edm::messagelogger;
  // ----------------------------------------------------------------------
  // Helper to construct the string->ELsev_ map on demand:
  // ----------------------------------------------------------------------

  typedef std::map<std::string const, ELseverityLevel::ELsev_, std::less<>> ELmap;

  ELmap const& loadMap() {
    static const ELmap m = {{ELzeroSeverity.getSymbol(), ELseverityLevel::ELsev_zeroSeverity},
                            {ELzeroSeverity.getName(), ELseverityLevel::ELsev_zeroSeverity},
                            {ELzeroSeverity.getInputStr(), ELseverityLevel::ELsev_zeroSeverity},
                            {ELzeroSeverity.getVarName(), ELseverityLevel::ELsev_zeroSeverity},
                            {ELdebug.getSymbol(), ELseverityLevel::ELsev_success},
                            {ELdebug.getName(), ELseverityLevel::ELsev_success},
                            {ELdebug.getInputStr(), ELseverityLevel::ELsev_success},
                            {ELdebug.getVarName(), ELseverityLevel::ELsev_success},
                            {ELinfo.getSymbol(), ELseverityLevel::ELsev_info},
                            {ELinfo.getName(), ELseverityLevel::ELsev_info},
                            {ELinfo.getInputStr(), ELseverityLevel::ELsev_info},
                            {ELinfo.getVarName(), ELseverityLevel::ELsev_info},
                            {ELfwkInfo.getSymbol(), ELseverityLevel::ELsev_fwkInfo},
                            {ELfwkInfo.getName(), ELseverityLevel::ELsev_fwkInfo},
                            {ELfwkInfo.getInputStr(), ELseverityLevel::ELsev_fwkInfo},
                            {ELfwkInfo.getVarName(), ELseverityLevel::ELsev_fwkInfo},
                            {ELwarning.getSymbol(), ELseverityLevel::ELsev_warning},
                            {ELwarning.getName(), ELseverityLevel::ELsev_warning},
                            {ELwarning.getInputStr(), ELseverityLevel::ELsev_warning},
                            {ELwarning.getVarName(), ELseverityLevel::ELsev_warning},
                            {ELerror.getSymbol(), ELseverityLevel::ELsev_error},
                            {ELerror.getName(), ELseverityLevel::ELsev_error},
                            {ELerror.getInputStr(), ELseverityLevel::ELsev_error},
                            {ELerror.getVarName(), ELseverityLevel::ELsev_error},
                            {ELunspecified.getSymbol(), ELseverityLevel::ELsev_unspecified},
                            {ELunspecified.getName(), ELseverityLevel::ELsev_unspecified},
                            {ELunspecified.getInputStr(), ELseverityLevel::ELsev_unspecified},
                            {ELunspecified.getVarName(), ELseverityLevel::ELsev_unspecified},
                            {ELsevere.getSymbol(), ELseverityLevel::ELsev_severe},
                            {ELsevere.getName(), ELseverityLevel::ELsev_severe},
                            {ELsevere.getInputStr(), ELseverityLevel::ELsev_severe},
                            {ELsevere.getVarName(), ELseverityLevel::ELsev_severe},
                            {ELhighestSeverity.getSymbol(), ELseverityLevel::ELsev_highestSeverity},
                            {ELhighestSeverity.getName(), ELseverityLevel::ELsev_highestSeverity},
                            {ELhighestSeverity.getInputStr(), ELseverityLevel::ELsev_highestSeverity},
                            {ELhighestSeverity.getVarName(), ELseverityLevel::ELsev_highestSeverity}};

    return m;
  }
}  // namespace

namespace edm::messagelogger {

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

}  // namespace edm::messagelogger
