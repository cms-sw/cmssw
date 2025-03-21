// -*- C++ -*-
//
// Package:     ParameterSet
// Class  :     ParameterDescriptionBase
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Thu Aug  2 15:35:43 EDT 2007
//

#include "FWCore/ParameterSet/interface/ParameterDescriptionBase.h"

#include "FWCore/ParameterSet/interface/DocFormatHelper.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include <iomanip>
#include <iostream>

namespace edm {

  ParameterDescriptionBase::ParameterDescriptionBase(
      std::string const& iLabel, ParameterTypes iType, bool isTracked, bool hasDefault, Comment const& iComment)
      : ParameterDescriptionNode(iComment),
        label_(iLabel),
        type_(iType),
        isTracked_(isTracked),
        hasDefault_(hasDefault) {
    if (label_.empty()) {
      throw Exception(errors::LogicError) << "Empty string used as a label for a parameter.  This is\n"
                                             "not allowed.\n";
    }
  }

  ParameterDescriptionBase::ParameterDescriptionBase(
      char const* iLabel, ParameterTypes iType, bool isTracked, bool hasDefault, Comment const& iComment)
      : ParameterDescriptionNode(iComment),
        label_(iLabel),
        type_(iType),
        isTracked_(isTracked),
        hasDefault_(hasDefault) {
    if (label_.empty()) {
      throw Exception(errors::LogicError) << "Empty string used as a label for a parameter.  This is\n"
                                             "not allowed.\n";
    }
  }

  ParameterDescriptionBase::~ParameterDescriptionBase() {}

  void ParameterDescriptionBase::throwParameterWrongTrackiness() const {
    std::string tr("a tracked");
    std::string shouldBe("untracked");
    if (isTracked()) {
      tr = "an untracked";
      shouldBe = "tracked";
    }

    throw Exception(errors::Configuration) << "In the configuration, parameter \"" << label()
                                           << "\" is defined "
                                              "as "
                                           << tr << " " << parameterTypeEnumToString(type()) << ".\n"
                                           << "It should be " << shouldBe << ".\n";
  }

  void ParameterDescriptionBase::throwParameterWrongType() const {
    std::string tr("an untracked");
    if (isTracked())
      tr = "a tracked";

    throw Exception(errors::Configuration) << "Parameter \"" << label()
                                           << "\" should be defined "
                                              "as "
                                           << tr << " " << parameterTypeEnumToString(type()) << ".\n"
                                           << "The type in the configuration is incorrect.\n";
  }

  void ParameterDescriptionBase::throwMissingRequiredNoDefault() const {
    std::string tr("untracked");
    if (isTracked())
      tr = "tracked";

    throw Exception(errors::Configuration)
        << "Missing required parameter.  It should have label \"" << label() << "\" and have type \"" << tr << " "
        << parameterTypeEnumToString(type()) << "\".\n"
        << "The description has no default.  The parameter must be defined "
           "in the configuration\n";
  }

  void ParameterDescriptionBase::checkAndGetLabelsAndTypes_(std::set<std::string>& usedLabels,
                                                            std::set<ParameterTypes>& parameterTypes,
                                                            std::set<ParameterTypes>& /*wildcardTypes*/) const {
    usedLabels.insert(label());
    parameterTypes.insert(type());
  }

  void ParameterDescriptionBase::validate_(ParameterSet& pset,
                                           std::set<std::string>& validatedLabels,
                                           Modifier modifier) const {
    bool exists = exists_(pset, isTracked());

    if (exists) {
      validatedLabels.insert(label());
    } else if (exists_(pset, !isTracked())) {
      throwParameterWrongTrackiness();
    } else if (pset.exists(label())) {
      throwParameterWrongType();
    }

    if (exists and modifier == Modifier::kObsolete) {
      edm::LogWarning("Configuration") << "ignoring obsolete parameter '" << label() << "'";
      return;
    }
    if (!exists && modifier == Modifier::kNone) {
      if (hasDefault()) {
        insertDefault_(pset);
        validatedLabels.insert(label());
      } else {
        throwMissingRequiredNoDefault();
      }
    }
  }

  void ParameterDescriptionBase::writeCfi_(std::ostream& os,
                                           Modifier modifier,
                                           bool& startWithComma,
                                           int indentation,
                                           CfiOptions& options,
                                           bool& wroteSomething) const {
    if (label().empty() or label()[0] == '@') {
      return;
    }

    auto check = cfi::needToSwitchToTyped(label(), options);
    if (check.first) {
      CfiOptions fullOp = cfi::Typed{};
      writeFullCfi(os, modifier, startWithComma, indentation, fullOp, wroteSomething);
    } else if (shouldWriteUntyped(options)) {
      if (modifier != Modifier::kObsolete) {
        writeLabelValueCfi(os, modifier == Modifier::kOptional, startWithComma, indentation, options, wroteSomething);
      }
    } else {
      writeFullCfi(os, modifier, startWithComma, indentation, options, wroteSomething);
    }
  }

  void ParameterDescriptionBase::writeLabelValueCfi(std::ostream& os,
                                                    bool optional,
                                                    bool& startWithComma,
                                                    int indentation,
                                                    CfiOptions& options,
                                                    bool& wroteSomething) const {
    constexpr std::string_view k_endList = "]";
    constexpr std::string_view k_endParenthesis = ")";
    constexpr std::string_view k_endBasicType = "";
    if (not hasDefault()) {
      return;
    }
    wroteSomething = true;
    if (startWithComma)
      os << ",";
    startWithComma = true;
    os << "\n";
    printSpaces(os, indentation);

    os << label() << " = ";
    std::string_view endDelimiter = k_endBasicType;
    switch (type()) {
      case k_vdouble:
      case k_vint32:
      case k_vint64:
      case k_vstringRaw:
      case k_vuint32:
      case k_vuint64:
      case k_VESInputTag:
      case k_VEventID:
      case k_VEventRange:
      case k_VInputTag:
      case k_VLuminosityBlockID:
      case k_VLuminosityBlockRange:
      case k_VPSet:
        os << "[";
        endDelimiter = k_endList;
        break;
      case k_PSet:
        os << "dict(";
        endDelimiter = k_endParenthesis;
        break;
      case k_EventID:
      case k_EventRange:
      case k_ESInputTag:
      case k_InputTag:
      case k_LuminosityBlockID:
      case k_LuminosityBlockRange:
        os << "(";
        endDelimiter = k_endParenthesis;
        break;
      default:
        break;
    }
    writeCfi_(os, indentation, options);
    os << endDelimiter;
    return;
  }

  void ParameterDescriptionBase::writeFullCfi(std::ostream& os,
                                              Modifier modifier,
                                              bool& startWithComma,
                                              int indentation,
                                              CfiOptions& options,
                                              bool& wroteSomething) const {
    wroteSomething = true;
    if (startWithComma)
      os << ",";
    startWithComma = true;

    os << "\n";
    printSpaces(os, indentation);

    os << label() << " = cms.";

    if (modifier == Modifier::kObsolete or !hasDefault()) {
      if (modifier == Modifier::kOptional) {
        os << "optional.";
      } else if (modifier == Modifier::kObsolete) {
        os << "obsolete.";
      } else {
        os << "required.";
      }
      if (!isTracked())
        os << "untracked.";
      os << parameterTypeEnumToString(type());
    } else {
      if (!isTracked())
        os << "untracked.";
      os << parameterTypeEnumToString(type()) << "(";
      writeCfi_(os, indentation, options);
      os << ")";
    }
  }
  void ParameterDescriptionBase::print_(std::ostream& os,
                                        Modifier modifier,
                                        bool writeToCfi,
                                        DocFormatHelper& dfh) const {
    const bool optional = (modifier == Modifier::kOptional);
    const bool obsolete = (modifier == Modifier::kObsolete);
    if (dfh.pass() == 0) {
      dfh.setAtLeast1(label().size());
      if (isTracked()) {
        dfh.setAtLeast2(parameterTypeEnumToString(type()).size());
      } else {
        dfh.setAtLeast2(parameterTypeEnumToString(type()).size() + 10U);
      }
      if (optional) {
        dfh.setAtLeast3(8U);
      }
    } else {
      if (dfh.brief()) {
        std::ios::fmtflags oldFlags = os.flags();

        dfh.indent(os);
        os << std::left << std::setw(dfh.column1()) << label() << " ";

        if (isTracked()) {
          os << std::setw(dfh.column2()) << parameterTypeEnumToString(type());
        } else {
          std::stringstream ss;
          ss << "untracked ";
          ss << parameterTypeEnumToString(type());
          os << std::setw(dfh.column2()) << ss.str();
        }
        os << " ";

        os << std::setw(dfh.column3());
        if (optional) {
          os << "optional";
        } else if (obsolete) {
          os << "obsolete";
        } else {
          os << "";
        }
        os << " ";
        os.flags(oldFlags);
        printDefault_(os, writeToCfi, dfh);
      } else {
        // not brief
        dfh.indent(os);
        os << label() << "\n";

        dfh.indent2(os);
        os << "type: ";
        if (!isTracked())
          os << "untracked ";

        os << parameterTypeEnumToString(type()) << " ";

        if (optional)
          os << "optional";
        if (obsolete)
          os << "obsolete";
        os << "\n";

        dfh.indent2(os);
        printDefault_(os, writeToCfi, dfh);

        if (!comment().empty()) {
          DocFormatHelper::wrapAndPrintText(os, comment(), dfh.startColumn2(), dfh.commentWidth());
        }
        os << "\n";
      }
    }
  }

  void ParameterDescriptionBase::printDefault_(std::ostream& os, bool writeToCfi, DocFormatHelper& dfh) const {
    if (!dfh.brief())
      os << "default: ";
    if (writeToCfi && hasDefault()) {
      if (hasNestedContent()) {
        os << "see Section " << dfh.section() << "." << dfh.counter();
      } else {
        if (dfh.brief()) {
          writeDoc_(os, dfh.indentation());
        } else {
          writeDoc_(os, dfh.startColumn2());
        }
      }
    } else if (!writeToCfi) {
      os << "none (do not write to cfi)";
    } else {
      os << "none";
    }
    os << "\n";
  }

  void ParameterDescriptionBase::printNestedContent_(std::ostream& os, bool /*optional*/, DocFormatHelper& dfh) const {
    int indentation = dfh.indentation();
    if (dfh.parent() != DocFormatHelper::TOP) {
      indentation -= DocFormatHelper::offsetSectionContent();
    }

    printSpaces(os, indentation);
    os << "Section " << dfh.section() << "." << dfh.counter() << " " << label() << " default contents: ";
    writeDoc_(os, indentation + 2);
    os << "\n";
    if (!dfh.brief())
      os << "\n";
  }

  bool ParameterDescriptionBase::partiallyExists_(ParameterSet const& pset) const { return exists(pset); }

  int ParameterDescriptionBase::howManyXORSubNodesExist_(ParameterSet const& pset) const {
    return exists(pset) ? 1 : 0;
  }
}  // namespace edm
