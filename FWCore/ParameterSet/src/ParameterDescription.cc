
#include "FWCore/ParameterSet/interface/ParameterDescription.h"

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/EventRange.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockRange.h"
#include "FWCore/ParameterSet/interface/DocFormatHelper.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/FillDescriptionFromPSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/VParameterSetEntry.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "boost/bind.hpp"

#include <cassert>
#include <cstdlib>
#include <iomanip>
#include <ostream>
#include <sstream>

namespace edm {

  // =================================================================

  ParameterDescription<ParameterSetDescription>::
  ParameterDescription(std::string const& iLabel,
                       ParameterSetDescription const& value,
                       bool isTracked) :
      ParameterDescriptionBase(iLabel, k_PSet, isTracked, true),
      psetDesc_(new ParameterSetDescription(value)) {
  }

  ParameterDescription<ParameterSetDescription>::
  ParameterDescription(char const* iLabel,
                       ParameterSetDescription const& value,
                       bool isTracked) :
      ParameterDescriptionBase(iLabel, k_PSet, isTracked, true),
      psetDesc_(new ParameterSetDescription(value)) {
  }

  ParameterDescription<ParameterSetDescription>::
  ~ParameterDescription() { }

  void
  ParameterDescription<ParameterSetDescription>::
  validate_(ParameterSet& pset,
           std::set<std::string>& validatedLabels,
           bool optional) const {

    bool exists = pset.existsAs<ParameterSet>(label(), isTracked());

    if(exists) {
      validatedLabels.insert(label());
    } else if(pset.existsAs<ParameterSet>(label(), !isTracked())) {
      throwParameterWrongTrackiness();
    } else if(pset.exists(label())) {
      throwParameterWrongType();
    }

    if(!optional && !exists) {
      if(isTracked()) {
        pset.addParameter(label(), ParameterSet());
      } else {
        pset.addUntrackedParameter(label(), ParameterSet());
      }
      validatedLabels.insert(label());
    }

    exists = pset.existsAs<ParameterSet>(label(), isTracked());

    if(exists) {
      ParameterSet * containedPSet = pset.getPSetForUpdate(label());
      psetDesc_->validate(*containedPSet);
    }
  }

  void
  ParameterDescription<ParameterSetDescription>::
  printDefault_(std::ostream& os,
                  bool writeToCfi,
                  DocFormatHelper& dfh) {
    os << "see Section " << dfh.section()
       << "." << dfh.counter();
    if(!writeToCfi) os << " (do not write to cfi)";
    os << "\n";
  }

  bool
  ParameterDescription<ParameterSetDescription>::
  hasNestedContent_() {
    return true;
  }

  void
  ParameterDescription<ParameterSetDescription>::
  printNestedContent_(std::ostream& os,
                      bool /*optional*/,
                      DocFormatHelper& dfh) {
    int indentation = dfh.indentation();
    if(dfh.parent() != DocFormatHelper::TOP) {
      indentation -= DocFormatHelper::offsetSectionContent();
    }

    std::stringstream ss;
    ss << dfh.section() << "." << dfh.counter();
    std::string newSection = ss.str();

    printSpaces(os, indentation);
    os << "Section " << newSection
       << " " << label() << " PSet description:\n";
    if(!dfh.brief()) os << "\n";

    DocFormatHelper new_dfh(dfh);
    new_dfh.init();
    new_dfh.setSection(newSection);
    if(dfh.parent() == DocFormatHelper::TOP) {
      new_dfh.setIndentation(indentation + DocFormatHelper::offsetSectionContent());
    }
    psetDesc_->print(os, new_dfh);
  }

  bool
  ParameterDescription<ParameterSetDescription>::
  exists_(ParameterSet const& pset) const {
    return pset.existsAs<ParameterSet>(label(), isTracked());
  }

  ParameterSetDescription const*
  ParameterDescription<ParameterSetDescription>::
  parameterSetDescription() const {
    return psetDesc_.operator->();
  }

  ParameterSetDescription *
  ParameterDescription<ParameterSetDescription>::
  parameterSetDescription() {
    return psetDesc_.operator->();
  }

  void
  ParameterDescription<ParameterSetDescription>::
  writeCfi_(std::ostream& os, int indentation) const {
    bool startWithComma = false;
    indentation += 2;
    psetDesc_->writeCfi(os, startWithComma, indentation);
  }

  void
  ParameterDescription<ParameterSetDescription>::
  writeDoc_(std::ostream&, int /*indentation*/) const {
  }

  // These next two should not be needed for this specialization
  bool
  ParameterDescription<ParameterSetDescription>::
  exists_(ParameterSet const&, bool /*isTracked*/) const {
    throw Exception(errors::LogicError);
    return true;
  }

  void
  ParameterDescription<ParameterSetDescription>::
  insertDefault_(ParameterSet&) const {
    throw Exception(errors::LogicError);
    return;
  }

  // =================================================================

  ParameterDescription<std::vector<ParameterSet> >::
  ParameterDescription(std::string const& iLabel,
                       ParameterSetDescription const& psetDesc,
                       bool isTracked,
                       std::vector<ParameterSet> const& vPset) :
      ParameterDescriptionBase(iLabel, k_VPSet, isTracked, true),
      psetDesc_(new ParameterSetDescription(psetDesc)),
      vPset_(vPset),
      partOfDefaultOfVPSet_(false) {
  }

  ParameterDescription<std::vector<ParameterSet> >::
  ParameterDescription(char const* iLabel,
                       ParameterSetDescription const& psetDesc,
                       bool isTracked,
                       std::vector<ParameterSet> const& vPset) :
      ParameterDescriptionBase(iLabel, k_VPSet, isTracked, true),
      psetDesc_(new ParameterSetDescription(psetDesc)),
      vPset_(vPset),
      partOfDefaultOfVPSet_(false) {
  }

  ParameterDescription<std::vector<ParameterSet> >::
  ParameterDescription(std::string const& iLabel,
                       ParameterSetDescription const& psetDesc,
                       bool isTracked) :
      ParameterDescriptionBase(iLabel, k_VPSet, isTracked, false),
      psetDesc_(new ParameterSetDescription(psetDesc)),
      vPset_(),
      partOfDefaultOfVPSet_(false) {
  }

  ParameterDescription<std::vector<ParameterSet> >::
  ParameterDescription(char const* iLabel,
                       ParameterSetDescription const& psetDesc,
                       bool isTracked) :
      ParameterDescriptionBase(iLabel, k_VPSet, isTracked, false),
      psetDesc_(new ParameterSetDescription(psetDesc)),
      vPset_(),
      partOfDefaultOfVPSet_(false) {
  }

  ParameterDescription<std::vector<ParameterSet> >::
  ~ParameterDescription() { }

  ParameterSetDescription const*
  ParameterDescription<std::vector<ParameterSet> >::
  parameterSetDescription() const {
    return psetDesc_.operator->();
  }

  ParameterSetDescription *
  ParameterDescription<std::vector<ParameterSet> >::
  parameterSetDescription() {
    return psetDesc_.operator->();
  }

  void
  ParameterDescription<std::vector<ParameterSet> >::
  validate_(ParameterSet& pset,
            std::set<std::string>& validatedLabels,
            bool optional) const {

    bool exists = pset.existsAs<std::vector<ParameterSet> >(label(), isTracked());

    if(exists) {
      validatedLabels.insert(label());
    } else if(pset.existsAs<std::vector<ParameterSet> >(label(), !isTracked())) {
      throwParameterWrongTrackiness();
    } else if(pset.exists(label())) {
      throwParameterWrongType();
    }

    if(!exists && !optional) {
      if(hasDefault()) {
        if(isTracked()) {
          pset.addParameter(label(), vPset_);
        } else {
          pset.addUntrackedParameter(label(), vPset_);
        }
        validatedLabels.insert(label());
      } else {
        throwMissingRequiredNoDefault();
      }
    }

    exists = pset.existsAs<std::vector<ParameterSet> >(label(), isTracked());
    if(exists) {
      VParameterSetEntry * vpsetEntry = pset.getPSetVectorForUpdate(label());
      assert(vpsetEntry);

      for(unsigned i = 0; i < vpsetEntry->size(); ++i) {
        psetDesc_->validate(vpsetEntry->psetInVector(i));
      }
    }
  }

  void
  ParameterDescription<std::vector<ParameterSet> >::
  printDefault_(std::ostream& os,
                bool writeToCfi,
                DocFormatHelper& dfh) {
    os << "see Section " << dfh.section()
       << "." << dfh.counter();
    if(!writeToCfi) os << " (do not write to cfi)";
    os << "\n";
  }


  bool
  ParameterDescription<std::vector<ParameterSet> >::
  hasNestedContent_() {
    return true;
  }

  void
  ParameterDescription<std::vector<ParameterSet> >::
  printNestedContent_(std::ostream& os,
                      bool /*optional*/,
                      DocFormatHelper& dfh) {

    int indentation = dfh.indentation();
    if(dfh.parent() != DocFormatHelper::TOP) {
      indentation -= DocFormatHelper::offsetSectionContent();
    }

    if(!partOfDefaultOfVPSet_) {
      printSpaces(os, indentation);
      os << "Section " << dfh.section() << "." << dfh.counter()
         << " " << label() << " VPSet description:\n";

      printSpaces(os, indentation + DocFormatHelper::offsetSectionContent());
      os << "All elements will be validated using the PSet description in Section "
         << dfh.section() << "." << dfh.counter() << ".1.\n";
    } else {
      printSpaces(os, indentation);
      os << "Section " << dfh.section() << "." << dfh.counter()
         << " " << " VPSet description for VPSet that is part of the default of a containing VPSet:\n";
    }

    printSpaces(os, indentation + DocFormatHelper::offsetSectionContent());

    unsigned subsectionOffset = 2;
    if(partOfDefaultOfVPSet_) subsectionOffset = 1;

    if(hasDefault()) {
      if(vPset_.size() == 0U) os << "The default VPSet is empty.\n";
      else if(vPset_.size() == 1U) os << "The default VPSet has 1 element.\n";
      else os << "The default VPSet has " << vPset_.size() << " elements.\n";

      if(vPset_.size() > 0U) {
        for(unsigned i = 0; i < vPset_.size(); ++i) {
          printSpaces(os, indentation + DocFormatHelper::offsetSectionContent());
          os << "[" << (i) << "]: see Section " << dfh.section()
             << "." << dfh.counter() << "." << (i + subsectionOffset) << "\n";
        }
      }
    } else {
      os << "Does not have a default VPSet.\n";
    }

    if(!dfh.brief()) os << "\n";

    if(!partOfDefaultOfVPSet_) {

      std::stringstream ss;
      ss << dfh.section() << "." << dfh.counter() << ".1";
      std::string newSection = ss.str();

      printSpaces(os, indentation);
      os << "Section " << newSection << " description of PSet used to validate elements of VPSet:\n";
      if(!dfh.brief()) os << "\n";

      DocFormatHelper new_dfh(dfh);
      new_dfh.init();
      new_dfh.setSection(newSection);
      if(dfh.parent() == DocFormatHelper::TOP) {
         new_dfh.setIndentation(indentation + DocFormatHelper::offsetSectionContent());
      }
      psetDesc_->print(os, new_dfh);
    }

    if(hasDefault()) {
      for(unsigned i = 0; i < vPset_.size(); ++i) {

        std::stringstream ss;
        ss << dfh.section() << "." << dfh.counter() << "." << (i + subsectionOffset);
        std::string newSection = ss.str();

        printSpaces(os, indentation);
        os << "Section " << newSection << " PSet description of "
           << "default VPSet element [" << i << "]\n";
        if(!dfh.brief()) os << "\n";

        DocFormatHelper new_dfh(dfh);
        new_dfh.init();
        new_dfh.setSection(newSection);
        if(dfh.parent() == DocFormatHelper::TOP) {
          new_dfh.setIndentation(indentation + DocFormatHelper::offsetSectionContent());
        }

        ParameterSetDescription defaultDescription;
        fillDescriptionFromPSet(vPset_[i], defaultDescription);
        defaultDescription.print(os, new_dfh);
      }
    }
  }

  bool
  ParameterDescription<std::vector<ParameterSet> >::
  exists_(ParameterSet const& pset) const {
    return pset.existsAs<std::vector<ParameterSet> >(label(), isTracked());
  }

  void
  ParameterDescription<std::vector<ParameterSet> >::
  writeOneElementToCfi(ParameterSet const& pset,
                       std::ostream& os,
                       int indentation,
                       bool& nextOneStartsWithAComma) {
    if(nextOneStartsWithAComma) os << ",";
    nextOneStartsWithAComma = true;
    os << "\n";
    printSpaces(os, indentation + 2);

    os << "cms.PSet(";

    bool startWithComma = false;
    int indent = indentation + 4;

    ParameterSetDescription psetDesc;
    fillDescriptionFromPSet(pset, psetDesc);
    psetDesc.writeCfi(os, startWithComma, indent);

    os << ")";
  }

  void
  ParameterDescription<std::vector<ParameterSet> >::
  writeCfi_(std::ostream& os, int indentation) const {
    bool nextOneStartsWithAComma = false;
    for_all(vPset_, boost::bind(&writeOneElementToCfi,
                                 _1,
                                 boost::ref(os),
                                 indentation,
                                 boost::ref(nextOneStartsWithAComma)));
    os << "\n";
    printSpaces(os, indentation);
  }

  void
  ParameterDescription<std::vector<ParameterSet> >::
  writeDoc_(std::ostream&, int /*indentation*/) const {
  }

  // These next two should not be needed for this specialization
  bool
  ParameterDescription<std::vector<ParameterSet> >::
  exists_(ParameterSet const&, bool /*isTracked*/) const {
    throw Exception(errors::LogicError);
    return true;
  }

  void
  ParameterDescription<std::vector<ParameterSet> >::
  insertDefault_(ParameterSet&) const {
    throw Exception(errors::LogicError);
    return;
  }

  // =================================================================

  namespace writeParameterValue {

    template<typename T>
    void writeSingleValue(std::ostream& os, T const& value, ValueFormat) {
      os << value;
    }

    // Specialize this for cases where the operator<< does not give
    // the proper formatting for a configuration file.

    // Formatting the doubles is a little tricky.  It is a requirement
    // that when a value of ***type double*** is added to a ParameterSetDescription
    // the EXACT same value of type double will be created and passed to the
    // ParameterSet after the cfi files have been read.  The tricky part
    // is these values usually appear in the C++ code and cfi file as text
    // strings (in decimal form).  We do our best to force the text
    // string in the C++ code to be the same as the text string in the
    // cfi file by judiciously rounding to smaller precision when possible.
    // But it is not always possible to force the text strings to be the
    // same.  Generally, there are differences when the text string in the
    // C++ code has many digits (probably more than a human will ever type).
    // Even in cases where the text strings differ, the values stored in
    // memory in variables of type double will be exactly the same.
    // The alternative to the approach here is to store the values as strings,
    // but that approach was rejected because it would require the
    // ParameterSetDescription to know how to parse the strings.
    void formatDouble(double value, std::string& result) {
      std::stringstream s;
      s << std::setprecision(17) << value;
      result = s.str();

      if(result.size() > 15 && std::string::npos != result.find(".")) {
        std::stringstream s;
        s << std::setprecision(15) << value;
        std::string resultLessPrecision = s.str();

        if(resultLessPrecision.size() < result.size() - 2) {
          double test = std::strtod(resultLessPrecision.c_str(), 0);
          if(test == value) {
            result = resultLessPrecision;
          }
        }
      }
    }

    template<>
    void writeSingleValue<double>(std::ostream& os, double const& value, ValueFormat) {
      std::string sValue;
      formatDouble(value, sValue);
      os << sValue;
    }

    template<>
    void writeSingleValue<bool>(std::ostream& os, bool const& value, ValueFormat) {
      value ? os << "True" : os << "False";
    }

    template<>
    void writeSingleValue<std::string>(std::ostream& os, std::string const& value, ValueFormat) {
      os << "'" << value << "'";
    }

    template<>
    void writeSingleValue<EventID>(std::ostream& os, EventID const& value, ValueFormat format) {
      if(format == CFI) {
        os << value.run() << ", " << value.luminosityBlock() << ", " << value.event();
      } else {
        if(value.luminosityBlock() == 0U) {
          os << value.run() << ":" << value.event();
        } else {
          os << value.run() << ":" << value.luminosityBlock() << ":" << value.event();
        }
      }
    }

    template<>
    void writeSingleValue<LuminosityBlockID>(std::ostream& os, LuminosityBlockID const& value, ValueFormat format) {
      if(format == CFI) os << value.run() << ", " << value.luminosityBlock();
      else os << value.run() << ":" << value.luminosityBlock();
    }

    template<>
    void writeSingleValue<EventRange>(std::ostream& os, EventRange const& value, ValueFormat format) {
      if(value.startLumi() == 0U) {
        if(format == CFI) os << "'" << value.startRun() << ":" << value.startEvent() << "-"
                                     << value.endRun() << ":" << value.endEvent() << "'";
        else os << value.startRun() << ":" << value.startEvent() << "-"
                << value.endRun() << ":" << value.endEvent();
      } else {
        if(format == CFI) os << "'" << value.startRun() << ":" << value.startLumi() << ":" << value.startEvent() << "-"
                                     << value.endRun() << ":" << value.endLumi() << ":" << value.endEvent() << "'";
        else os << value.startRun() << ":" << value.startLumi() << ":" << value.startEvent() << "-"
                << value.endRun() << ":" << value.endLumi() << ":" << value.endEvent();
      }
    }

    template<>
    void writeSingleValue<LuminosityBlockRange>(std::ostream& os, LuminosityBlockRange const& value, ValueFormat format) {
      if(format == CFI) os << "'" << value.startRun() << ":" << value.startLumi() << "-"
                            << value.endRun() << ":" << value.endLumi() << "'";
      else os << value.startRun() << ":" << value.startLumi() << "-"
              << value.endRun() << ":" << value.endLumi();
    }

    template<>
    void writeSingleValue<InputTag>(std::ostream& os, InputTag const& value, ValueFormat format) {
      if(format == CFI) {
        os << "'" << value.label() << "'";
        if(!value.instance().empty() || !value.process().empty()) {
          os << ", '" << value.instance() << "'";
        }
        if(!value.process().empty()) {
          os << ", '" << value.process() << "'";
        }
      } else {
        os << "'" << value.label();
        if(!value.instance().empty() || !value.process().empty()) {
          os << ":" << value.instance();
        }
        if(!value.process().empty()) {
          os << ":" << value.process();
        }
        os << "'";
      }
    }

    template<>
    void writeSingleValue<FileInPath>(std::ostream& os, FileInPath const& value, ValueFormat) {
      os << "'" << value.relativePath() << "'";
    }

    template<typename T>
    void writeValue(std::ostream& os, T const& value_, ValueFormat format) {
      std::ios_base::fmtflags ff = os.flags(std::ios_base::dec);
      os.width(0);
      writeSingleValue<T>(os, value_, format);
      os.flags(ff);
    }

    template<typename T>
    void writeValueInVector(std::ostream& os, T const& value, ValueFormat format) {
      writeSingleValue<T>(os, value, format);
    }

    // Specializations for cases where we write the single values into
    // vectors differently than when there is only one not in a vector.
    template<>
    void writeValueInVector<EventID>(std::ostream& os, EventID const& value, ValueFormat format) {
      if(value.luminosityBlock() == 0U) {
        if(format == CFI) os << "'" << value.run() << ":" << value.event() << "'";
        else os << value.run() << ":" << value.event();
      } else {
        if(format == CFI) os << "'" << value.run() << ":" << value.luminosityBlock() << ":" << value.event() << "'";
        else os << value.run() << ":" << value.luminosityBlock() << ":" << value.event();
      }
    }

    template<>
    void writeValueInVector<LuminosityBlockID>(std::ostream& os, LuminosityBlockID const& value, ValueFormat format) {
      if(format == CFI) os << "'" << value.run() << ":" << value.luminosityBlock() << "'";
      else os << value.run() << ":" << value.luminosityBlock();
    }

    template<>
    void writeValueInVector<InputTag>(std::ostream& os, InputTag const& value, ValueFormat) {
      os << "'" << value.label();
      if(!value.instance().empty() || !value.process().empty()) {
        os << ":" << value.instance();
      }
      if(!value.process().empty()) {
        os << ":" << value.process();
      }
      os << "'";
    }

    template<typename T>
    void writeValueInVectorWithSpace(T const& value,
                                     std::ostream& os,
                                     int indentation,
                                     bool& startWithComma,
                                     ValueFormat format,
                                     int& i) {
      if(startWithComma && format == CFI) os << ",";
      startWithComma = true;
      os << "\n" << std::setw(indentation) << "";
      if(format == DOC) os << "[" << i << "]: ";
      writeValueInVector<T>(os, value, format);
      ++i;
    }

    template<typename T>
    void writeVector(std::ostream& os, int indentation, std::vector<T> const& value_, ValueFormat format) {
      std::ios_base::fmtflags ff = os.flags(std::ios_base::dec);
      char oldFill = os.fill();
      os.width(0);
      if(value_.size() == 0U && format == DOC) {
        os << "empty";
      } else if(value_.size() == 1U && format == CFI) {
        writeValueInVector<T>(os, value_[0], format);
      } else if(value_.size() >= 1U) {
        if(format == DOC) os << "(vector size = " << value_.size() << ")";
        os.fill(' ');
        bool startWithComma = false;
        int i = 0;
        for_all(value_, boost::bind(&writeValueInVectorWithSpace<T>,
                                    _1,
                                    boost::ref(os),
                                    indentation + 2,
                                    boost::ref(startWithComma),
                                    format,
                                    boost::ref(i)));
        if(format == CFI) os << "\n" << std::setw(indentation) << "";
      }
      os.flags(ff);
      os.fill(oldFill);
    }

    void writeValue(std::ostream& os, int, int const& value_, ValueFormat format) {
      writeValue<int>(os, value_, format);
    }

    void writeValue(std::ostream& os, int indentation, std::vector<int> const& value_, ValueFormat format) {
      writeVector<int>(os, indentation, value_, format);
    }

    void writeValue(std::ostream& os, int, unsigned const& value_, ValueFormat format) {
      writeValue<unsigned>(os, value_, format);
    }

    void writeValue(std::ostream& os, int indentation, std::vector<unsigned> const& value_, ValueFormat format) {
      writeVector<unsigned>(os, indentation, value_, format);
    }

    void writeValue(std::ostream& os, int, long long const& value_, ValueFormat format) {
      writeValue<long long>(os, value_, format);
    }

    void writeValue(std::ostream& os, int indentation, std::vector<long long> const& value_, ValueFormat format) {
      writeVector<long long>(os, indentation, value_, format);
    }

    void writeValue(std::ostream& os, int, unsigned long long const& value_, ValueFormat format) {
      writeValue<unsigned long long>(os, value_, format);
    }

    void writeValue(std::ostream& os, int indentation, std::vector<unsigned long long> const& value_, ValueFormat format) {
      writeVector<unsigned long long>(os, indentation, value_, format);
    }

    void writeValue(std::ostream& os, int, double const& value_, ValueFormat format) {
      writeValue<double>(os, value_, format);
    }

    void writeValue(std::ostream& os, int indentation, std::vector<double> const& value_, ValueFormat format) {
      writeVector<double>(os, indentation, value_, format);
    }

    void writeValue(std::ostream& os, int, bool const& value_, ValueFormat format) {
      writeValue<bool>(os, value_, format);
    }

    void writeValue(std::ostream& os, int, std::string const& value_, ValueFormat format) {
      writeValue<std::string>(os, value_, format);
    }

    void writeValue(std::ostream& os, int indentation, std::vector<std::string> const& value_, ValueFormat format) {
      writeVector<std::string>(os, indentation, value_, format);
    }

    void writeValue(std::ostream& os, int, EventID const& value_, ValueFormat format) {
      writeValue<EventID>(os, value_, format);
    }

    void writeValue(std::ostream& os, int indentation, std::vector<EventID> const& value_, ValueFormat format) {
      writeVector<EventID>(os, indentation, value_, format);
    }

    void writeValue(std::ostream& os, int, LuminosityBlockID const& value_, ValueFormat format) {
      writeValue<LuminosityBlockID>(os, value_, format);
    }

    void writeValue(std::ostream& os, int indentation, std::vector<LuminosityBlockID> const& value_, ValueFormat format) {
      writeVector<LuminosityBlockID>(os, indentation, value_, format);
    }

    void writeValue(std::ostream& os, int, LuminosityBlockRange const& value_, ValueFormat format) {
      writeValue<LuminosityBlockRange>(os, value_, format);
    }

    void writeValue(std::ostream& os, int indentation, std::vector<LuminosityBlockRange> const& value_, ValueFormat format) {
      writeVector<LuminosityBlockRange>(os, indentation, value_, format);
    }

    void writeValue(std::ostream& os, int, EventRange const& value_, ValueFormat format) {
      writeValue<EventRange>(os, value_, format);
    }

    void writeValue(std::ostream& os, int indentation, std::vector<EventRange> const& value_, ValueFormat format) {
      writeVector<EventRange>(os, indentation, value_, format);
    }

    void writeValue(std::ostream& os, int, InputTag const& value_, ValueFormat format) {
      writeValue<InputTag>(os, value_, format);
    }

    void writeValue(std::ostream& os, int indentation, std::vector<InputTag> const& value_, ValueFormat format) {
      writeVector<InputTag>(os, indentation, value_, format);
    }

    void writeValue(std::ostream& os, int, FileInPath const& value_, ValueFormat format) {
      writeValue<FileInPath>(os, value_, format);
    }

    bool hasNestedContent(int const&) { return false; }
    bool hasNestedContent(std::vector<int> const& value) { return value.size() > 5U; }
    bool hasNestedContent(unsigned const&) { return false; }
    bool hasNestedContent(std::vector<unsigned> const& value) { return value.size() > 5U; }
    bool hasNestedContent(long long const&) { return false; }
    bool hasNestedContent(std::vector<long long> const& value) { return value.size() > 5U; }
    bool hasNestedContent(unsigned long long const&) { return false; }
    bool hasNestedContent(std::vector<unsigned long long> const& value) { return value.size() > 5U; }
    bool hasNestedContent(double const&) { return false; }
    bool hasNestedContent(std::vector<double> const& value) { return value.size() > 5U; }
    bool hasNestedContent(bool const&) { return false; }
    bool hasNestedContent(std::string const&) { return false; }
    bool hasNestedContent(std::vector<std::string> const& value) { return value.size() > 5U; }
    bool hasNestedContent(EventID const&) { return false; }
    bool hasNestedContent(std::vector<EventID> const& value) { return value.size() > 5U; }
    bool hasNestedContent(LuminosityBlockID const&) { return false; }
    bool hasNestedContent(std::vector<LuminosityBlockID> const& value) { return value.size() > 5U; }
    bool hasNestedContent(LuminosityBlockRange const&) { return false; }
    bool hasNestedContent(std::vector<LuminosityBlockRange> const& value) { return value.size() > 5U; }
    bool hasNestedContent(EventRange const&) { return false; }
    bool hasNestedContent(std::vector<EventRange> const& value) { return value.size() > 5U; }
    bool hasNestedContent(InputTag const&) { return false; }
    bool hasNestedContent(std::vector<InputTag> const& value) { return value.size() > 5U; }
    bool hasNestedContent(FileInPath const&) { return false; }
  }
}
