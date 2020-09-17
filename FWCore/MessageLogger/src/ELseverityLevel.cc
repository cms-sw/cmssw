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

namespace edm {

  // ----------------------------------------------------------------------
  // Helper to construct the string->ELsev_ map on demand:
  // ----------------------------------------------------------------------

  typedef std::map<std::string const, ELseverityLevel::ELsev_, std::less<>> ELmap;

  static ELmap const& loadMap() {
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

  // ----------------------------------------------------------------------
  // Birth/death:
  // ----------------------------------------------------------------------

  ELseverityLevel::ELseverityLevel(enum ELsev_ lev) : myLevel(lev) {
#ifdef ELsevConTRACE
    std::cerr << "--- ELseverityLevel " << lev << " (" << getName() << ")\n" << std::flush;
#endif
  }

  ELseverityLevel::ELseverityLevel(std::string_view s) {
    static ELmap const& m = loadMap();

    ELmap::const_iterator i = m.find(s);
    myLevel = (i == m.end()) ? ELsev_unspecified : i->second;
  }

  ELseverityLevel::~ELseverityLevel() { ; }

  // ----------------------------------------------------------------------
  // Comparator:
  // ----------------------------------------------------------------------

  int ELseverityLevel::cmp(ELseverityLevel const& e) const { return myLevel - e.myLevel; }

  // ----------------------------------------------------------------------
  // Accessors:
  // ----------------------------------------------------------------------

  int ELseverityLevel::getLevel() const { return myLevel; }

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

  // ----------------------------------------------------------------------
  // Declare the globally available severity objects,
  // one generator function and one proxy per non-default ELsev_:
  // ----------------------------------------------------------------------

  ELseverityLevel const ELzeroSeverityGen() {
    static ELseverityLevel const e(ELseverityLevel::ELsev_zeroSeverity);
    return e;
  }
  ELslProxy<ELzeroSeverityGen> const ELzeroSeverity;

  ELseverityLevel const ELdebugGen() {
    static ELseverityLevel const e(ELseverityLevel::ELsev_success);
    return e;
  }
  ELslProxy<ELdebugGen> const ELdebug;

  ELseverityLevel const ELinfoGen() {
    static ELseverityLevel const e(ELseverityLevel::ELsev_info);
    return e;
  }
  ELslProxy<ELinfoGen> const ELinfo;

  ELseverityLevel const ELwarningGen() {
    static ELseverityLevel const e(ELseverityLevel::ELsev_warning);
    return e;
  }
  ELslProxy<ELwarningGen> const ELwarning;

  ELseverityLevel const ELerrorGen() {
    static ELseverityLevel const e(ELseverityLevel::ELsev_error);
    return e;
  }
  ELslProxy<ELerrorGen> const ELerror;

  ELseverityLevel const ELunspecifiedGen() {
    static ELseverityLevel const e(ELseverityLevel::ELsev_unspecified);
    return e;
  }
  ELslProxy<ELunspecifiedGen> const ELunspecified;

  ELseverityLevel const ELsevereGen() {
    static ELseverityLevel const e(ELseverityLevel::ELsev_severe);
    return e;
  }
  ELslProxy<ELsevereGen> const ELsevere;

  ELseverityLevel const ELhighestSeverityGen() {
    static ELseverityLevel const e(ELseverityLevel::ELsev_highestSeverity);
    return e;
  }
  ELslProxy<ELhighestSeverityGen> const ELhighestSeverity;

  // ----------------------------------------------------------------------

}  // end of namespace edm  */
