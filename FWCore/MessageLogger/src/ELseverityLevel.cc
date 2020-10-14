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
// 29-Jun-1998 mf       Created file.
// 26-Aug-1998 WEB      Made ELseverityLevel object less weighty.
// 16-Jun-1999 mf       Added constructor from string, plus two lists
//                      of names to match.  Also added default constructor,
//                      more streamlined than default lev on original.
// 23-Jun-1999 mf       Modifications to properly handle pre-main order
//                      of initialization issues:
//                              Instantiation ofthe 14 const ELseverity &'s
//                              Instantiation of objectsInitialized as false
//                              Constructor of ELinitializeGlobalSeverityObjects
//                              Removed guarantor function in favor of the
//                              constructor.
// 30-Jun-1999 mf       Modifications to eliminate propblems with order of
//                      globals initializations:
//                              Constructor from lev calls translate()
//                              Constructor from string uses translate()
//                              translate() method
//                              List of strings for names in side getname() etc.
//                              Immediate initilization of ELsevLevGlobals
//                              Mods involving ELinitializeGlobalSeverityObjects
// 12-Jun-2000 web      Final fix to global static initialization problem
// 27-Jun-2000 web      Fix order-of-static-destruction problem
// 24-Aug-2000 web      Fix defective C++ switch generation
// 13-Jun-2007 mf       Change (requested by CMS) the name Severe to System
//			(since that his how MessageLogger uses that level)
// 21-Apr-2009 mf	Change the symbol for ELsev_success (which is used
//                      by CMS for LogDebug) from -! to -d.
// ----------------------------------------------------------------------

#include <cassert>
#include <ostream>

#include "FWCore/MessageLogger/interface/ELseverityLevel.h"
#include "FWCore/MessageLogger/interface/ELmap.h"

// Possible Traces
// #define ELsevConTRACE
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
      ret[ELsev_success] = "-d";  // 4/21/09 mf
      ret[ELsev_info] = "-i";
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
      ret[ELsev_success] = "Debug";  // 4/21/09 mf
      ret[ELsev_info] = "Info";
      ret[ELsev_warning] = "Warning";
      ret[ELsev_error] = "Error";
      ret[ELsev_unspecified] = "??";
      ret[ELsev_severe] = "System";  // 6/13/07 mf
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
      ret[ELsev_warning] = "WARNING";
      ret[ELsev_error] = "ERROR";
      ret[ELsev_unspecified] = "UNSPECIFIED";
      ret[ELsev_severe] = "SYSTEM";  // 6/13/07 mf
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
      ret[ELsev_success] = "ELdebug          ";  // 4/21/09
      ret[ELsev_info] = "ELinfo           ";
      ret[ELsev_warning] = "ELwarning        ";
      ret[ELsev_error] = "ELerror          ";
      ret[ELsev_unspecified] = "ELunspecified    ";
      ret[ELsev_severe] = "ELsystem         ";  // 6/13/07
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
