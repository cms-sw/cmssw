// -*- C++ -*-
//
// Package:     Services
// Class  :     MessageServicePSetValidation
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  M. Fischler
//         Created:  Wed May 20 2009
//
//

// system include files

#include <algorithm>

// user include files

#include "FWCore/MessageService/src/MessageServicePSetValidation.h"

using namespace edm;
using namespace edm::service;

namespace edm {
  namespace service {

    std::string edm::service::MessageServicePSetValidation::operator()(ParameterSet const& pset) {
      messageLoggerPSet(pset);
      return flaws_.str();
    }  // operator() to validate the PSet passed in

    void edm::service::MessageServicePSetValidation::messageLoggerPSet(ParameterSet const& pset) {
      // See if new config API is being used
      if (pset.exists("files") or
          (not(pset.exists("destinations") or pset.existsAs<std::vector<std::string>>("statistics", true) or
               pset.existsAs<std::vector<std::string>>("statistics", false) or pset.exists("categories")))) {
        return;
      }

      // Four types of material are allowed at the MessageLogger level:
      //   PSet lists (such as destinations or categories
      //   Suppression lists, such as SuppressInfo or debugModules
      //   General parameters, such as threshold or messageSummaryToJobReport
      //   Nested PSets, such as those for each destination

      // PSet lists

      psetLists(pset);

      // Suppression lists

      suppressionLists(pset);

      // No other vstrings

      vStringsCheck(pset, "MessageLogger");

      // General Parameters

      check<bool>(pset, "MessageLogger", "messageSummaryToJobReport");
      std::string dumps = check<std::string>(pset, "MessageLogger", "generate_preconfiguration_message");
      std::string thresh = check<std::string>(pset, "MessageLogger", "threshold");
      if (!thresh.empty())
        validateThreshold(thresh, "MessageLogger");
      check<unsigned int>(pset, "MessageLogger", "waiting_threshold");

      // Nested PSets

      destinationPSets(pset);
      defaultPSet(pset);
      statisticsPSets(pset);
      categoryPSets(pset, "MessageLogger");

      // No other PSets -- unless they contain optionalPSet or placeholder=True

      noOtherPsets(pset);

      // Nothing else -- look for int, unsigned int, bool, float, double, string

      noneExcept<int>(pset, "MessageLogger", "int");
      noneExcept<unsigned int>(pset, "MessageLogger", "unsigned int", "waiting_threshold");
      noneExcept<bool>(pset, "MessageLogger", "bool", "messageSummaryToJobReport");
      // Note - at this, the upper MessageLogger PSet level, the use of
      // optionalPSet makes no sense, so we are OK letting that be a flaw
      noneExcept<float>(pset, "MessageLogger", "float");
      noneExcept<double>(pset, "MessageLogger", "double");
      noneExcept<std::string>(pset, "MessageLogger", "string", "threshold", "generate_preconfiguration_message");

      // Append explanatory information if flaws were found

      if (!flaws_.str().empty()) {
        flaws_ << "\nThe above are from MessageLogger configuration validation.\n"
               << "In most cases, these involve lines that the logger configuration code\n"
               << "would not process, but which the cfg creator obviously meant to have "
               << "effect.\n";
      }

    }  // messageLoggerPSet

    void edm::service::MessageServicePSetValidation::psetLists(ParameterSet const& pset) {
      destinations_ = check<vString>(pset, "MessageLogger", "destinations");
      noDuplicates(destinations_, "MessageLogger", "destinations");
      noKeywords(destinations_, "MessageLogger", "destinations");
      noNonPSetUsage(pset, destinations_, "MessageLogger", "destinations");
      // REMOVED: noCoutCerrClash(destinations_,"MessageLogger", "destinations");

      statistics_ = check<vString>(pset, "MessageLogger", "statistics");
      noDuplicates(statistics_, "MessageLogger", "statistics");
      noKeywords(statistics_, "MessageLogger", "statistics");
      noNonPSetUsage(pset, statistics_, "MessageLogger", "statistics");

      categories_ = check<vString>(pset, "MessageLogger", "categories");
      noDuplicates(categories_, "MessageLogger", "categories");
      noKeywords(categories_, "MessageLogger", "categories");
      noNonPSetUsage(pset, categories_, "MessageLogger", "categories");
      noDuplicates(categories_, destinations_, "MessageLogger", "categories", "destinations");
      noDuplicates(categories_, statistics_, "MessageLogger", "categories", "statistics");

    }  // psetLists

    void edm::service::MessageServicePSetValidation::suppressionLists(ParameterSet const& pset) {
      debugModules_ = check<vString>(pset, "MessageLogger", "debugModules");
      bool dmStar = wildcard(debugModules_);
      if (dmStar && debugModules_.size() != 1) {
        flaws_ << "MessageLogger"
               << " PSet: \n"
               << "debugModules contains wildcard character *"
               << " and also " << debugModules_.size() - 1 << " other entries - * must be alone\n";
      }
      suppressDebug_ = check<vString>(pset, "MessageLogger", "suppressDebug");
      if ((!suppressDebug_.empty()) && (!dmStar)) {
        flaws_ << "MessageLogger"
               << " PSet: \n"
               << "suppressDebug contains modules, but debugModules is not *\n"
               << "Unless all the debugModules are enabled,\n"
               << "suppressing specific modules is meaningless\n";
      }
      if (wildcard(suppressDebug_)) {
        flaws_ << "MessageLogger"
               << " PSet: \n"
               << "Use of wildcard (*) in suppressDebug is not supported\n"
               << "By default, LogDebug is suppressed for all modules\n";
      }
      suppressInfo_ = check<vString>(pset, "MessageLogger", "suppressInfo");
      if (wildcard(suppressInfo_)) {
        flaws_ << "MessageLogger"
               << " PSet: \n"
               << "Use of wildcard (*) in suppressInfo is not supported\n";
      }
      suppressFwkInfo_ = check<vString>(pset, "MessageLogger", "suppressFwkInfo");
      if (wildcard(suppressFwkInfo_)) {
        flaws_ << "MessageLogger"
               << " PSet: \n"
               << "Use of wildcard (*) in suppressFwkInfo is not supported\n";
      }
      suppressWarning_ = check<vString>(pset, "MessageLogger", "suppressWarning");
      if (wildcard(suppressWarning_)) {
        flaws_ << "MessageLogger"
               << " PSet: \n"
               << "Use of wildcard (*) in suppressWarning is not supported\n";
      }
      suppressError_ = check<vString>(pset, "MessageLogger", "suppressError");
      if (wildcard(suppressError_)) {
        flaws_ << "MessageLogger"
               << " PSet: \n"
               << "Use of wildcard (*) in suppressError is not supported\n";
      }

    }  // suppressionLists

    void edm::service::MessageServicePSetValidation::vStringsCheck(ParameterSet const& pset,
                                                                   std::string const& /*psetName*/) {
      vString vStrings = pset.getParameterNamesForType<vString>(false);
      vString::const_iterator end = vStrings.end();
      for (vString::const_iterator i = vStrings.begin(); i != end; ++i) {
        if (!allowedVstring(*i)) {
          flaws_ << "MessageLogger"
                 << " PSet: \n"
                 << (*i) << " is used as a vstring, "
                 << "but no such vstring is recognized\n";
        }
      }
      vStrings = pset.getParameterNamesForType<vString>(true);
      end = vStrings.end();
      for (vString::const_iterator i = vStrings.begin(); i != end; ++i) {
        flaws_ << "MessageLogger"
               << " PSet: \n"
               << (*i) << " is used as a tracked vstring: "
               << "tracked parameters not allowed here\n";
      }
    }  // vStringsCheck

    bool edm::service::MessageServicePSetValidation::allowedVstring(std::string const& s) {
      if (s == "destinations")
        return true;
      if (s == "statistics")
        return true;
      if (s == "destinations")
        return true;
      if (s == "categories")
        return true;
      if (s == "debugModules")
        return true;
      if (s == "suppressInfo")
        return true;
      if (s == "suppressFwkInfo")
        return true;
      if (s == "suppressDebug")
        return true;
      if (s == "suppressWarning")
        return true;
      if (s == "suppressError")
        return true;
      return false;
    }  // allowedVstring

    bool edm::service::MessageServicePSetValidation::validateThreshold(std::string const& thresh,
                                                                       std::string const& psetName) {
      if (checkThreshold(thresh))
        return true;
      flaws_ << psetName << " PSet: \n"
             << "threshold has value " << thresh << " which is not among {DEBUG, INFO, FWKINFO, WARNING, ERROR}\n";
      return false;
    }  // validateThreshold

    bool edm::service::MessageServicePSetValidation::checkThreshold(std::string const& thresh) {
      if (thresh == "WARNING")
        return true;
      if (thresh == "INFO")
        return true;
      if (thresh == "FWKINFO")
        return true;
      if (thresh == "ERROR")
        return true;
      if (thresh == "DEBUG")
        return true;
      return false;
    }

    void edm::service::MessageServicePSetValidation::noDuplicates(vString const& v,
                                                                  std::string const& psetName,
                                                                  std::string const& parameterLabel) {
      vString::const_iterator end = v.end();
      for (vString::const_iterator i = v.begin(); i != end; ++i) {
        for (vString::const_iterator j = i + 1; j != end; ++j) {
          if (*i == *j) {
            flaws_ << psetName << " PSet: \n"
                   << "in vString " << parameterLabel << " duplication of the string " << *i << "\n";
          }
        }
      }
    }  // noDuplicates(v)

    void edm::service::MessageServicePSetValidation::noDuplicates(vString const& v1,
                                                                  vString const& v2,
                                                                  std::string const& psetName,
                                                                  std::string const& p1,
                                                                  std::string const& p2) {
      vString::const_iterator end1 = v1.end();
      vString::const_iterator end2 = v2.end();
      for (vString::const_iterator i = v1.begin(); i != end1; ++i) {
        for (vString::const_iterator j = v2.begin(); j != end2; ++j) {
          if (*i == *j) {
            flaws_ << psetName << " PSet: \n"
                   << "in vStrings " << p1 << " and " << p2 << " duplication of the string " << *i << "\n";
          }
        }
      }
    }  // noDuplicates(v1,v2)

    void edm::service::MessageServicePSetValidation::noCoutCerrClash(vString const& v,
                                                                     std::string const& psetName,
                                                                     std::string const& parameterLabel) {
      vString::const_iterator end = v.end();
      bool coutPresent = false;
      bool cerrPresent = false;
      for (vString::const_iterator i = v.begin(); i != end; ++i) {
        if (*i == "cout")
          coutPresent = true;
        if (*i == "cerr")
          cerrPresent = true;
      }
      if (coutPresent && cerrPresent) {
        flaws_ << psetName << " PSet: \n"
               << "vString " << parameterLabel << " has both cout and cerr \n";
      }
    }  // noCoutCerrClash(v)

    void edm::service::MessageServicePSetValidation::noKeywords(vString const& v,
                                                                std::string const& psetName,
                                                                std::string const& parameterLabel) {
      vString::const_iterator end = v.end();
      for (vString::const_iterator i = v.begin(); i != end; ++i) {
        if (!keywordCheck(*i)) {
          flaws_ << psetName << " PSet: \n"
                 << "vString " << parameterLabel << " should not contain the keyword " << *i << "\n";
        }
      }
    }  // noKeywords(v)

    bool edm::service::MessageServicePSetValidation::keywordCheck(std::string const& word) {
      if (word == "default")
        return false;
      if (word == "categories")
        return false;
      if (word == "destinations")
        return false;
      if (word == "statistics")
        return false;
      if (word == "debugModules")
        return false;
      if (word == "suppressInfo")
        return false;
      if (word == "suppressFwkInfo")
        return false;
      if (word == "suppressDebug")
        return false;
      if (word == "suppressWarning")
        return false;
      if (word == "suppressError")
        return false;
      if (word == "threshold")
        return false;
      if (word == "ERROR")
        return false;
      if (word == "WARNING")
        return false;
      if (word == "FWKINFO")
        return false;
      if (word == "INFO")
        return false;
      if (word == "DEBUG")
        return false;
      if (word == "placeholder")
        return false;
      if (word == "limit")
        return false;
      if (word == "reportEvery")
        return false;
      if (word == "timespan")
        return false;
      if (word == "noLineBreaks")
        return false;
      if (word == "lineLength")
        return false;
      if (word == "noTimeStamps")
        return false;
      if (word == "output")
        return false;
      if (word == "filename")
        return false;
      if (word == "extension")
        return false;
      if (word == "reset")
        return false;
      if (word == "optionalPSet")
        return false;
      if (word == "enableStatistics")
        return false;
      if (word == "statisticsThreshold")
        return false;
      if (word == "resetStatistics")
        return false;
      return true;
    }  // keywordCheck

    void edm::service::MessageServicePSetValidation::noNonPSetUsage(ParameterSet const& pset,
                                                                    vString const& v,
                                                                    std::string const& psetName,
                                                                    std::string const& parameterLabel) {
      disallowedParam<int>(pset, v, psetName, parameterLabel, "int");
      disallowedParam<unsigned int>(pset, v, psetName, parameterLabel, "uint");
      disallowedParam<bool>(pset, v, psetName, parameterLabel, "bool");
      disallowedParam<float>(pset, v, psetName, parameterLabel, "float");
      disallowedParam<double>(pset, v, psetName, parameterLabel, "double");
      disallowedParam<std::string>(pset, v, psetName, parameterLabel, "string");
      disallowedParam<std::vector<std::string>>(pset, v, psetName, parameterLabel, "vstring");
    }  // noNonPSetUsage

    void edm::service::MessageServicePSetValidation::noBadParams(vString const& v,
                                                                 vString const& params,
                                                                 std::string const& psetName,
                                                                 std::string const& parameterLabel,
                                                                 std::string const& type) {
      vString::const_iterator end1 = v.end();
      vString::const_iterator end2 = params.end();
      for (vString::const_iterator i = v.begin(); i != end1; ++i) {
        for (vString::const_iterator j = params.begin(); j != end2; ++j) {
          if (*i == *j) {
            flaws_ << psetName << " PSet: \n"
                   << *i << " (listed in vstring " << parameterLabel << ")\n"
                   << "is used as a parameter of type " << type << " instead of as a PSet \n";
          }
        }
      }

    }  // noBadParams

    bool edm::service::MessageServicePSetValidation::wildcard(vString const& v) {
      vString::const_iterator end = v.end();
      for (vString::const_iterator i = v.begin(); i != end; ++i) {
        if ((*i) == "*")
          return true;
      }
      return false;
    }

    void edm::service::MessageServicePSetValidation::noOtherPsets(ParameterSet const& pset) {
      vString psnames;
      pset.getParameterSetNames(psnames, false);
      vString::const_iterator end = psnames.end();
      for (vString::const_iterator i = psnames.begin(); i != end; ++i) {
        if (lookForMatch(destinations_, *i))
          continue;
        if (lookForMatch(statistics_, *i))
          continue;
        if (lookForMatch(categories_, *i))
          continue;
        if ((*i) == "default")
          continue;
        ParameterSet empty_PSet;
        bool ok_optionalPSet = false;
        try {
          ParameterSet const& culprit = pset.getUntrackedParameterSet((*i), empty_PSet);
          ok_optionalPSet = culprit.getUntrackedParameter<bool>("placeholder", ok_optionalPSet);
          ok_optionalPSet = culprit.getUntrackedParameter<bool>("optionalPSet", ok_optionalPSet);
        } catch (cms::Exception& e) {
        }
        if (ok_optionalPSet)
          continue;
        flaws_ << "MessageLogger "
               << " PSet: \n"
               << *i << " is an unrecognized name for a PSet\n";
      }
      psnames.clear();
      unsigned int n = pset.getParameterSetNames(psnames, true);
      if (n > 0) {
        end = psnames.end();
        for (vString::const_iterator i = psnames.begin(); i != end; ++i) {
          flaws_ << "MessageLogger "
                 << " PSet: \n"
                 << "PSet " << *i << " is tracked - not allowed\n";
        }
      }
    }

    bool edm::service::MessageServicePSetValidation::lookForMatch(vString const& v, std::string const& s) {
      vString::const_iterator begin = v.begin();
      vString::const_iterator end = v.end();
      return (std::find(begin, end, s) != end);
    }

    void edm::service::MessageServicePSetValidation::destinationPSets(ParameterSet const& pset) {
      ParameterSet empty_PSet;
      std::vector<std::string>::const_iterator end = destinations_.end();
      for (std::vector<std::string>::const_iterator i = destinations_.begin(); i != end; ++i) {
        ParameterSet const& d = pset.getUntrackedParameterSet(*i, empty_PSet);
        destinationPSet(d, *i);
      }
    }  // destinationPSets

    void edm::service::MessageServicePSetValidation::destinationPSet(ParameterSet const& pset,
                                                                     std::string const& psetName) {
      // Category PSets

      categoryPSets(pset, psetName);

      // No other PSets -- unless they contain optionalPSet or placeholder=True

      noNoncategoryPsets(pset, psetName);

      // General parameters

      check<bool>(pset, psetName, "placeholder");
      {
        std::string thresh = check<std::string>(pset, "psetName", "statisticsThreshold");
        if (!thresh.empty())
          validateThreshold(thresh, psetName);
      }
      std::string thresh = check<std::string>(pset, "psetName", "threshold");
      if (!thresh.empty())
        validateThreshold(thresh, psetName);
      check<bool>(pset, psetName, "noLineBreaks");
      check<int>(pset, psetName, "lineLength");
      check<bool>(pset, psetName, "noTimeStamps");
      check<bool>(pset, psetName, "enableStatistics");
      check<bool>(pset, psetName, "resetStatistics");
      std::string s = check<std::string>(pset, "psetName", "filename");
      if ((s == "cerr") || (s == "cout")) {
        flaws_ << psetName << " PSet: \n" << s << " is not allowed as a value of filename \n";
      }
      s = check<std::string>(pset, "psetName", "extension");
      if ((s == "cerr") || (s == "cout")) {
        flaws_ << psetName << " PSet: \n" << s << " is not allowed as a value of extension \n";
      }
      s = check<std::string>(pset, "psetName", "output");

      // No other parameters

      noneExcept<int>(pset, psetName, "int", "lineLength");

      vString okbool;
      okbool.push_back("placeholder");
      okbool.push_back("optionalPSet");
      okbool.push_back("noLineBreaks");
      okbool.push_back("noTimeStamps");
      okbool.push_back("enableStatistics");
      okbool.push_back("resetStatistics");
      noneExcept<bool>(pset, psetName, "bool", okbool);
      vString okstring;
      okstring.push_back("statisticsThreshold");
      okstring.push_back("threshold");
      okstring.push_back("output");
      okstring.push_back("filename");
      okstring.push_back("extension");
      noneExcept<std::string>(pset, psetName, "string", okstring);

    }  // destinationPSet

    void edm::service::MessageServicePSetValidation::defaultPSet(ParameterSet const& main_pset) {
      ParameterSet empty_PSet;
      ParameterSet const& pset = main_pset.getUntrackedParameterSet("default", empty_PSet);
      std::string psetName = "default (at MessageLogger main level)";

      // Category PSets

      categoryPSets(pset, psetName);

      // No other PSets -- unless they contain optionalPSet or placeholder=True

      noNoncategoryPsets(pset, psetName);

      // Parameters applying to the default category

      catInts(pset, psetName, "default");

      // General parameters

      check<bool>(pset, psetName, "placeholder");
      std::string thresh = check<std::string>(pset, "psetName", "threshold");
      if (!thresh.empty())
        validateThreshold(thresh, psetName);
      check<bool>(pset, psetName, "noLineBreaks");
      check<int>(pset, psetName, "limit");
      check<int>(pset, psetName, "reportEvery");
      check<int>(pset, psetName, "timespan");
      check<int>(pset, psetName, "lineLength");
      check<bool>(pset, psetName, "noTimeStamps");

      // No other parameters
      vString okint;
      okint.push_back("limit");
      okint.push_back("reportEvery");
      okint.push_back("timespan");
      okint.push_back("lineLength");
      noneExcept<int>(pset, psetName, "int", okint);
      vString okbool;
      okbool.push_back("placeholder");
      okbool.push_back("optionalPSet");
      okbool.push_back("noLineBreaks");
      okbool.push_back("noTimeStamps");
      noneExcept<bool>(pset, psetName, "bool", okbool);
      vString okstring;
      okstring.push_back("threshold");
      noneExcept<std::string>(pset, psetName, "string", okstring);

    }  // defaultPSet

    void edm::service::MessageServicePSetValidation::statisticsPSets(ParameterSet const& pset) {
      ParameterSet empty_PSet;
      std::vector<std::string>::const_iterator end = statistics_.end();
      for (std::vector<std::string>::const_iterator i = statistics_.begin(); i != end; ++i) {
        if (lookForMatch(destinations_, *i))
          continue;
        ParameterSet const& d = pset.getUntrackedParameterSet(*i, empty_PSet);
        statisticsPSet(d, *i);
      }
    }  // statisticsPSets

    void edm::service::MessageServicePSetValidation::statisticsPSet(ParameterSet const& pset,
                                                                    std::string const& psetName) {
      // Category PSets

      categoryPSets(pset, psetName);

      // No other PSets -- unless they contain optionalPSet or placeholder=True

      noNoncategoryPsets(pset, psetName);

      // General parameters

      std::string thresh = check<std::string>(pset, "psetName", "threshold");
      if (!thresh.empty())
        validateThreshold(thresh, psetName);
      check<bool>(pset, psetName, "placeholder");
      check<bool>(pset, psetName, "reset");
      std::string s = check<std::string>(pset, "psetName", "filename");
      if ((s == "cerr") || (s == "cout")) {
        flaws_ << psetName << " PSet: \n" << s << " is not allowed as a value of filename \n";
      }
      s = check<std::string>(pset, "psetName", "extension");
      if ((s == "cerr") || (s == "cout")) {
        flaws_ << psetName << " PSet: \n" << s << " is not allowed as a value of extension \n";
      }
      s = check<std::string>(pset, "psetName", "output");

      // No other parameters

      noneExcept<int>(pset, psetName, "int");

      vString okbool;
      okbool.push_back("placeholder");
      okbool.push_back("optionalPSet");
      okbool.push_back("reset");
      noneExcept<bool>(pset, psetName, "bool", okbool);
      vString okstring;
      okstring.push_back("output");
      okstring.push_back("filename");
      okstring.push_back("extension");
      okstring.push_back("threshold");
      noneExcept<std::string>(pset, psetName, "string", okstring);

    }  // statisticsPSet

    void edm::service::MessageServicePSetValidation::noNoncategoryPsets(ParameterSet const& pset,
                                                                        std::string const& psetName) {
      vString psnames;
      pset.getParameterSetNames(psnames, false);
      vString::const_iterator end = psnames.end();
      for (vString::const_iterator i = psnames.begin(); i != end; ++i) {
        if (lookForMatch(categories_, *i))
          continue;
        if ((*i) == "default")
          continue;
        if ((*i) == "ERROR")
          continue;
        if ((*i) == "WARNING")
          continue;
        if ((*i) == "FWKINFO")
          continue;
        if ((*i) == "INFO")
          continue;
        if ((*i) == "DEBUG")
          continue;
        ParameterSet empty_PSet;
        bool ok_optionalPSet = false;
        try {
          ParameterSet const& culprit = pset.getUntrackedParameterSet((*i), empty_PSet);
          ok_optionalPSet = culprit.getUntrackedParameter<bool>("placeholder", ok_optionalPSet);
          ok_optionalPSet = culprit.getUntrackedParameter<bool>("optionalPSet", ok_optionalPSet);
        } catch (cms::Exception& e) {
        }
        if (ok_optionalPSet)
          continue;
        flaws_ << psetName << " PSet: \n" << *i << " is an unrecognized name for a PSet in this context \n";
      }
      psnames.clear();
      unsigned int n = pset.getParameterSetNames(psnames, true);
      if (n > 0) {
        end = psnames.end();
        for (vString::const_iterator i = psnames.begin(); i != end; ++i) {
          flaws_ << psetName << " PSet: \n"
                 << "PSet " << *i << " is tracked - not allowed\n";
        }
      }
    }  // noNoncategoryPsets

    void edm::service::MessageServicePSetValidation::categoryPSets(ParameterSet const& pset,
                                                                   std::string const& psetName) {
      categoryPSet(pset, psetName, "ERROR");
      categoryPSet(pset, psetName, "WARNING");
      categoryPSet(pset, psetName, "FWKINFO");
      categoryPSet(pset, psetName, "INFO");
      categoryPSet(pset, psetName, "DEBUG");
      if (psetName != "MessageLogger")
        categoryPSet(pset, psetName, "default");
      // The above conditional is because default in the main level is treated
      // as a set of defaults differnt from those of a simple category.
      std::vector<std::string>::const_iterator end = categories_.end();
      for (std::vector<std::string>::const_iterator i = categories_.begin(); i != end; ++i) {
        categoryPSet(pset, psetName, *i);
      }
    }  // categoryPSets

    void edm::service::MessageServicePSetValidation::categoryPSet(ParameterSet const& pset,
                                                                  std::string const& OuterPsetName,
                                                                  std::string const& categoryName) {
      if (pset.existsAs<ParameterSet>(categoryName, true)) {
        flaws_ << OuterPsetName << " PSet: \n"
               << "Category PSet " << categoryName << " is tracked - not allowed\n";
        return;
      }
      ParameterSet empty_PSet;
      ParameterSet const& c = pset.getUntrackedParameterSet(categoryName, empty_PSet);
      std::string const& psetName(OuterPsetName);
      catInts(c, psetName, categoryName);
      catNone<unsigned int>(c, psetName, categoryName, "unsigned int");
      catBoolRestriction(c, psetName, categoryName, "bool");
      catNone<float>(c, psetName, categoryName, "float");
      catNone<double>(c, psetName, categoryName, "double");
      catNone<std::string>(c, psetName, categoryName, "string");
      catNone<vString>(c, psetName, categoryName, "vSting");
      catNoPSets(c, psetName, categoryName);
    }  // categoryPSet

    void edm::service::MessageServicePSetValidation::catInts(ParameterSet const& pset,
                                                             std::string const& psetName,
                                                             std::string const& categoryName) {
      vString x = pset.getParameterNamesForType<int>(false);
      vString::const_iterator end = x.end();
      for (vString::const_iterator i = x.begin(); i != end; ++i) {
        if (*i == "limit")
          continue;
        if (*i == "reportEvery")
          continue;
        if (*i == "timespan")
          continue;
        flaws_ << categoryName << " category PSet nested in " << psetName << " PSet: \n"
               << (*i) << " is not an allowed parameter within a category PSet \n";
      }
      x = pset.getParameterNamesForType<int>(true);
      end = x.end();
      for (vString::const_iterator i = x.begin(); i != end; ++i) {
        flaws_ << categoryName << " category PSet nested in " << psetName << " PSet: \n"
               << (*i) << " is used as a tracked int \n"
               << "Tracked parameters not allowed here \n";
      }
    }  // catInts()

    void edm::service::MessageServicePSetValidation::catNoPSets(ParameterSet const& pset,
                                                                std::string const& psetName,
                                                                std::string const& categoryName) {
      vString psnames;
      pset.getParameterSetNames(psnames, false);
      vString::const_iterator end = psnames.end();
      for (vString::const_iterator i = psnames.begin(); i != end; ++i) {
        flaws_ << categoryName << " category PSet nested in " << psetName << " PSet: \n"
               << *i << " is used as a  PSet\n"
               << "PSets not allowed within a category PSet\n";
      }
      psnames.clear();
      unsigned int n = pset.getParameterSetNames(psnames, true);
      if (n > 0) {
        end = psnames.end();
        for (vString::const_iterator i = psnames.begin(); i != end; ++i) {
          flaws_ << categoryName << " category PSet nested in " << psetName << " PSet: \n"
                 << *i << " is used as a tracked PSet\n"
                 << "tracked parameters not permitted, and "
                 << "PSets not allowed within a category PSet\n";
        }
      }
    }  // catNoPSets

    void edm::service::MessageServicePSetValidation::catBoolRestriction(ParameterSet const& pset,
                                                                        std::string const& psetName,
                                                                        std::string const& categoryName,
                                                                        std::string const& type) {
      vString x = pset.getParameterNamesForType<bool>(false);
      vString::const_iterator end = x.end();
      for (vString::const_iterator i = x.begin(); i != end; ++i) {
        if (((*i) == "placeholder") || ((*i) == "optionalPSet"))
          continue;
        flaws_ << categoryName << " category PSet nested in " << psetName << " PSet: \n"
               << (*i) << " is used as a " << type << "\n"
               << "Usage of " << type << " is not recognized here\n";
      }
      x = pset.getParameterNamesForType<bool>(true);
      end = x.end();
      for (vString::const_iterator i = x.begin(); i != end; ++i) {
        flaws_ << categoryName << " category PSet nested in " << psetName << " PSet: \n"
               << (*i) << " is used as a tracked " << type << "\n"
               << "Tracked parameters not allowed here, "
               << " and even untracked it would not be recognized\n";
      }
    }  // catBoolRestriction()

  }  // end of namespace service
}  // end of namespace edm
