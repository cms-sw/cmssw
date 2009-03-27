
#include "FWCore/ParameterSet/interface/ParameterDescription.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/VParameterSetEntry.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/EventRange.h"
#include "DataFormats/Provenance/interface/LuminosityBlockRange.h"

#include "boost/bind.hpp"

#include <ostream>
#include <iomanip>
#include <sstream>
#include <cstdlib>

namespace edm {

  // =================================================================

  ParameterDescription<ParameterSetDescription>::
  ParameterDescription(std::string const& iLabel,
                       ParameterSetDescription const& value,
                       bool isTracked
		       ) :
      ParameterDescriptionBase(iLabel, k_PSet, isTracked),
      psetDesc_(new ParameterSetDescription(value)) {
  }

  ParameterDescription<ParameterSetDescription>::
  ParameterDescription(char const* iLabel,
                       ParameterSetDescription const& value,
                       bool isTracked
     		      ) :
      ParameterDescriptionBase(iLabel, k_PSet, isTracked),
      psetDesc_(new ParameterSetDescription(value)) {
  }

  ParameterDescription<ParameterSetDescription>::
  ~ParameterDescription() { }

  void
  ParameterDescription<ParameterSetDescription>::
  validate_(ParameterSet & pset,
           std::set<std::string> & validatedLabels,
           bool optional) const {

    bool exists = pset.existsAs<ParameterSet>(label(), isTracked());

    if (exists) {
      validatedLabels.insert(label());
    }
    else if (pset.existsAs<ParameterSet>(label(), !isTracked())) {
      throwParameterWrongTrackiness();
    }
    else if (pset.exists(label())) {
      throwParameterWrongType();
    }

    if (!optional && !exists) {
      if (isTracked()) {
        pset.addParameter(label(), ParameterSet());
      }
      else {
        pset.addUntrackedParameter(label(), ParameterSet());
      }
      validatedLabels.insert(label());
    }

    exists = pset.existsAs<ParameterSet>(label(), isTracked());

    if (exists) {
      ParameterSet * containedPSet = pset.getPSetForUpdate(label());
      psetDesc_->validate(*containedPSet);
    }
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
  writeCfi_(std::ostream & os, int indentation) const {
    bool startWithComma = false;
    indentation += 2;
    psetDesc_->writeCfi(os, startWithComma, indentation);
  }

  // =================================================================

  ParameterDescription<std::vector<ParameterSetDescription> >::
  ParameterDescription(std::string const& iLabel,
                       std::vector<ParameterSetDescription> const& value,
                       bool isTracked
		      ) :
      ParameterDescriptionBase(iLabel, k_VPSet, isTracked),
      vPsetDesc_(new std::vector<ParameterSetDescription>(value)) {
  }

  ParameterDescription<std::vector<ParameterSetDescription> >::
  ParameterDescription(char const* iLabel,
                       std::vector<ParameterSetDescription> const& value,
                       bool isTracked
		      ) :
      ParameterDescriptionBase(iLabel, k_VPSet, isTracked),
      vPsetDesc_(new std::vector<ParameterSetDescription>(value)) {
  }

  ParameterDescription<std::vector<ParameterSetDescription> >::
  ~ParameterDescription() { }

  void
  ParameterDescription<std::vector<ParameterSetDescription> >::
  validate_(ParameterSet & pset,
            std::set<std::string> & validatedLabels,
            bool optional) const {

    bool exists = pset.existsAs<std::vector<ParameterSet> >(label(), isTracked());

    if (exists) {
      validatedLabels.insert(label());
    }
    else if (pset.existsAs<std::vector<ParameterSet> >(label(), !isTracked())) {
      throwParameterWrongTrackiness();
    }
    else if (pset.exists(label())) {
      throwParameterWrongType();
    }

    if (!optional && !exists) {
      std::vector<ParameterSet> vpset(vPsetDesc_->size(), ParameterSet());
      if (isTracked()) {
        pset.addParameter(label(), vpset);
      }
      else {
        pset.addUntrackedParameter(label(), vpset);
      }
      validatedLabels.insert(label());
    }

    exists = pset.existsAs<std::vector<ParameterSet> >(label(), isTracked());
    if (exists) {
      VParameterSetEntry * vpsetEntry = pset.getPSetVectorForUpdate(label());
      assert(vpsetEntry);
      if (vpsetEntry->size() != vPsetDesc_->size()) {
        throw edm::Exception(errors::Configuration)
          << "Unexpected number of ParameterSets in vector of parameter sets named \"" << label() << "\".";
      }
      int i = 0;
      for_all(*vPsetDesc_,
              boost::bind(&ParameterDescription<std::vector<ParameterSetDescription> >::validateDescription,
                          boost::cref(this),
                          _1,
                          vpsetEntry,
                          boost::ref(i)));
    }
  }

  bool
  ParameterDescription<std::vector<ParameterSetDescription> >::
  exists_(ParameterSet const& pset) const {
    return pset.existsAs<std::vector<ParameterSet> >(label(), isTracked());
  }

  void
  ParameterDescription<std::vector<ParameterSetDescription> >::
  validateDescription(ParameterSetDescription const& psetDescription,
                      VParameterSetEntry * vpsetEntry,
                      int & i) const {
    psetDescription.validate(vpsetEntry->psetInVector(i));
    ++i;
  }

  std::vector<ParameterSetDescription> const*
  ParameterDescription<std::vector<ParameterSetDescription> >::
  parameterSetDescriptions() const {
    return vPsetDesc_.operator->();
  }

  std::vector<ParameterSetDescription> *
  ParameterDescription<std::vector<ParameterSetDescription> >::
  parameterSetDescriptions() {
    return vPsetDesc_.operator->();
  }

  void
  ParameterDescription<std::vector<ParameterSetDescription> >::
  writeOneDescriptionToCfi(ParameterSetDescription const& psetDesc,
                           std::ostream & os,
                           int indentation,
                           bool & nextOneStartsWithAComma) {
    if (nextOneStartsWithAComma) os << ",";
    nextOneStartsWithAComma = true;
    os << "\n" << std::setw(indentation + 2) << " ";
    os << "cms.PSet(";

    bool startWithComma = false;
    int indent = indentation + 4;
    psetDesc.writeCfi(os, startWithComma, indent);

    os << ")";
  }

  void
  ParameterDescription<std::vector<ParameterSetDescription> >::
  writeCfi_(std::ostream & os, int indentation) const {
    bool nextOneStartsWithAComma = false;
    for_all(*vPsetDesc_, boost::bind(&writeOneDescriptionToCfi,
                                    _1,
                                    boost::ref(os),
                                    indentation,
                                    boost::ref(nextOneStartsWithAComma)));
    os << "\n" << std::setw(indentation) << " ";
  }

  // =================================================================

  namespace writeToCfi {

    template <class T>
    void writeSingleValue(std::ostream & os, T const& value) {
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
    void formatDouble(double value, std::string & result)
    {
      std::stringstream s;
      s << std::setprecision(17) << value;
      result = s.str();

      if (result.size() > 15 && std::string::npos != result.find(".")) {
        std::stringstream s;
        s << std::setprecision(15) << value;
        std::string resultLessPrecision = s.str();

        if (resultLessPrecision.size() < result.size() - 2) {
          double test = std::strtod(resultLessPrecision.c_str(), 0);
          if (test == value) {
            result = resultLessPrecision;
          }
        }
      }
    }

    template <>
    void writeSingleValue<double>(std::ostream & os, double const& value) {
      std::string sValue;
      formatDouble(value, sValue);
      os << sValue;
    }

    template <>
    void writeSingleValue<bool>(std::ostream & os, bool const& value) {
      value ? os << "True" : os << "False";
    }

    template <>
    void writeSingleValue<std::string>(std::ostream & os, std::string const& value) {
      os << "'" << value << "'";
    }

    template <>
    void writeSingleValue<edm::EventID>(std::ostream & os, edm::EventID const& value) {
      os << value.run() << ", " << value.event();
    }

    template <>
    void writeSingleValue<edm::LuminosityBlockID>(std::ostream & os, edm::LuminosityBlockID const& value) {
      os << value.run() << ", " << value.luminosityBlock();
    }

    template <>
    void writeSingleValue<edm::InputTag>(std::ostream & os, edm::InputTag const& value) {
      os << "'" << value.label() << "'";
      if (!value.instance().empty() || !value.process().empty()) {
        os << ", '" << value.instance() << "'";
      }
      if (!value.process().empty()) {
        os << ", '" << value.process() << "'";
      }
    }

    template <>
    void writeSingleValue<edm::FileInPath>(std::ostream & os, edm::FileInPath const& value) {
      os << "'" << value.relativePath() << "'";
    }

    template <class T>
    void writeValueToCfi(std::ostream & os, T const& value_) {
      std::ios_base::fmtflags ff = os.flags(std::ios_base::dec);
      os.width(0);
      writeSingleValue<T>(os, value_);
      os.flags(ff);
    }

    template <class T>
    void writeValueInVector(std::ostream & os, T const& value) {
      writeSingleValue<T>(os, value);
    }

    // Specializations for cases where we write the single values into
    // vectors differently than when there is only one not in a vector.
    template <>
    void writeValueInVector<edm::EventID>(std::ostream & os, edm::EventID const& value) {
      os << "'" << value.run() << ":" << value.event() << "'";
    }

    template <>
    void writeValueInVector<edm::LuminosityBlockID>(std::ostream & os, edm::LuminosityBlockID const& value) {
      os << "'" << value.run() << ":" << value.luminosityBlock() << "'";
    }

    template <>
    void writeValueInVector<edm::InputTag>(std::ostream & os, edm::InputTag const& value) {
      os << "'" << value.label();
      if (!value.instance().empty() || !value.process().empty()) {
        os << ":" << value.instance();
      }
      if (!value.process().empty()) {
        os << ":" << value.process();
      }
      os << "'";
    }

    template <class T>
    void writeValueInVectorWithSpace(T const& value,
                                     std::ostream & os,
                                     int indentation,
                                     bool & startWithComma) {
      if (startWithComma) os << ",";
      startWithComma = true;
      os << "\n" << std::setw(indentation) << "";
      writeValueInVector<T>(os, value);
    }

    template <class T>
    void writeVectorToCfi(std::ostream & os, int indentation, std::vector<T> const& value_) {
      std::ios_base::fmtflags ff = os.flags(std::ios_base::dec);
      os.width(0);
      if (value_.size() == 1U) {
        writeValueInVector<T>(os, value_[0]);
      }
      else if (value_.size() > 1U) {
        os.fill(' ');
        bool startWithComma = false;
        for_all(value_, boost::bind(&writeValueInVectorWithSpace<T>,
                                    _1,
                                    boost::ref(os),
                                    indentation + 2,
                                    boost::ref(startWithComma)));
        os << "\n" << std::setw(indentation) << "";
      }
      os.flags(ff);
    }

    void writeValueToCfi(std::ostream & os, int indentation, int const& value_) {
      writeValueToCfi<int>(os, value_);
    }

    void writeValueToCfi(std::ostream & os, int indentation, std::vector<int> const& value_) {
      writeVectorToCfi<int>(os, indentation, value_);
    }

    void writeValueToCfi(std::ostream & os, int indentation, unsigned const& value_) {
      writeValueToCfi<unsigned>(os, value_);
    }

    void writeValueToCfi(std::ostream & os, int indentation, std::vector<unsigned> const& value_) {
      writeVectorToCfi<unsigned>(os, indentation, value_);
    }

    void writeValueToCfi(std::ostream & os, int indentation, boost::int64_t const& value_) {
      writeValueToCfi<boost::int64_t>(os, value_);
    }

    void writeValueToCfi(std::ostream & os, int indentation, std::vector<boost::int64_t> const& value_) {
      writeVectorToCfi<boost::int64_t>(os, indentation, value_);
    }

    void writeValueToCfi(std::ostream & os, int indentation, boost::uint64_t const& value_) {
      writeValueToCfi<boost::uint64_t>(os, value_);
    }

    void writeValueToCfi(std::ostream & os, int indentation, std::vector<boost::uint64_t> const& value_) {
      writeVectorToCfi<boost::uint64_t>(os, indentation, value_);
    }

    void writeValueToCfi(std::ostream & os, int indentation, double const& value_) {
      writeValueToCfi<double>(os, value_);
    }

    void writeValueToCfi(std::ostream & os, int indentation, std::vector<double> const& value_) {
      writeVectorToCfi<double>(os, indentation, value_);
    }

    void writeValueToCfi(std::ostream & os, int indentation, bool const& value_) {
      writeValueToCfi<bool>(os, value_);
    }

    void writeValueToCfi(std::ostream & os, int indentation, std::string const& value_) {
      writeValueToCfi<std::string>(os, value_);
    }

    void writeValueToCfi(std::ostream & os, int indentation, std::vector<std::string> const& value_) {
      writeVectorToCfi<std::string>(os, indentation, value_);
    }

    void writeValueToCfi(std::ostream & os, int indentation, edm::EventID const& value_) {
      writeValueToCfi<edm::EventID>(os, value_);
    }

    void writeValueToCfi(std::ostream & os, int indentation, std::vector<edm::EventID> const& value_) {
      writeVectorToCfi<edm::EventID>(os, indentation, value_);
    }

    void writeValueToCfi(std::ostream & os, int indentation, edm::LuminosityBlockID const& value_) {
      writeValueToCfi<edm::LuminosityBlockID>(os, value_);
    }

    void writeValueToCfi(std::ostream & os, int indentation, std::vector<edm::LuminosityBlockID> const& value_) {
      writeVectorToCfi<edm::LuminosityBlockID>(os, indentation, value_);
    }

    void writeValueToCfi(std::ostream & os, int indentation, edm::LuminosityBlockRange const& value_) {
      writeValueToCfi<edm::LuminosityBlockRange>(os, value_);
    }

    void writeValueToCfi(std::ostream & os, int indentation, std::vector<edm::LuminosityBlockRange> const& value_) {
      writeVectorToCfi<edm::LuminosityBlockRange>(os, indentation, value_);
    }

    void writeValueToCfi(std::ostream & os, int indentation, edm::EventRange const& value_) {
      writeValueToCfi<edm::EventRange>(os, value_);
    }

    void writeValueToCfi(std::ostream & os, int indentation, std::vector<edm::EventRange> const& value_) {
      writeVectorToCfi<edm::EventRange>(os, indentation, value_);
    }

    void writeValueToCfi(std::ostream & os, int indentation, edm::InputTag const& value_) {
      writeValueToCfi<edm::InputTag>(os, value_);
    }

    void writeValueToCfi(std::ostream & os, int indentation, std::vector<edm::InputTag> const& value_) {
      writeVectorToCfi<edm::InputTag>(os, indentation, value_);
    }

    void writeValueToCfi(std::ostream & os, int indentation, edm::FileInPath const& value_) {
      writeValueToCfi<edm::FileInPath>(os, value_);
    }
  }
}
