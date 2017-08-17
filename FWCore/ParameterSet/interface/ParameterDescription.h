#ifndef FWCore_ParameterSet_ParameterDescription_h
#define FWCore_ParameterSet_ParameterDescription_h
// -*- C++ -*-
//
// Package:     ParameterSet
// Class  :     ParameterDescription
//
/**\class ParameterDescription ParameterDescription.h FWCore/ParameterSet/interface/ParameterDescription.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Aug  2 15:33:51 EDT 2007
//

#include "FWCore/ParameterSet/interface/ParameterDescriptionBase.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/value_ptr.h"

#include <string>
#include <vector>
#include <iosfwd>
#include <set>

namespace edm {

  class ParameterSetDescription;

  class EventID;
  class LuminosityBlockID;
  class LuminosityBlockRange;
  class EventRange;
  class InputTag;
  class FileInPath;
  class DocFormatHelper;

  namespace writeParameterValue {

    enum ValueFormat { CFI, DOC };

    void writeValue(std::ostream& os, int indentation, int const& value_, ValueFormat format);
    void writeValue(std::ostream& os, int indentation, std::vector<int> const& value_, ValueFormat format);
    void writeValue(std::ostream& os, int indentation, unsigned const& value_, ValueFormat format);
    void writeValue(std::ostream& os, int indentation, std::vector<unsigned> const& value_, ValueFormat format);
    void writeValue(std::ostream& os, int indentation, long long const& value_, ValueFormat format);
    void writeValue(std::ostream& os, int indentation, std::vector<long long> const& value_, ValueFormat format);
    void writeValue(std::ostream& os, int indentation, unsigned long long const& value_, ValueFormat format);
    void writeValue(std::ostream& os, int indentation, std::vector<unsigned long long> const& value_, ValueFormat format);
    void writeValue(std::ostream& os, int indentation, double const& value_, ValueFormat format);
    void writeValue(std::ostream& os, int indentation, std::vector<double> const& value_, ValueFormat format);
    void writeValue(std::ostream& os, int indentation, bool const& value_, ValueFormat format);
    void writeValue(std::ostream& os, int indentation, std::string const& value_, ValueFormat format);
    void writeValue(std::ostream& os, int indentation, std::vector<std::string> const& value_, ValueFormat format);
    void writeValue(std::ostream& os, int indentation, EventID const& value_, ValueFormat format);
    void writeValue(std::ostream& os, int indentation, std::vector<EventID> const& value_, ValueFormat format);
    void writeValue(std::ostream& os, int indentation, LuminosityBlockID const& value_, ValueFormat format);
    void writeValue(std::ostream& os, int indentation, std::vector<LuminosityBlockID> const& value_, ValueFormat format);
    void writeValue(std::ostream& os, int indentation, LuminosityBlockRange const& value_, ValueFormat format);
    void writeValue(std::ostream& os, int indentation, std::vector<LuminosityBlockRange> const& value_, ValueFormat format);
    void writeValue(std::ostream& os, int indentation, EventRange const& value_, ValueFormat format);
    void writeValue(std::ostream& os, int indentation, std::vector<EventRange> const& value_, ValueFormat format);
    void writeValue(std::ostream& os, int indentation, InputTag const& value_, ValueFormat format);
    void writeValue(std::ostream& os, int indentation, std::vector<InputTag> const& value_, ValueFormat format);
    void writeValue(std::ostream& os, int indentation, FileInPath const& value_, ValueFormat format);

    bool hasNestedContent(int const& value);
    bool hasNestedContent(std::vector<int> const& value);
    bool hasNestedContent(unsigned const& value);
    bool hasNestedContent(std::vector<unsigned> const& value);
    bool hasNestedContent(long long const& value);
    bool hasNestedContent(std::vector<long long> const& value);
    bool hasNestedContent(unsigned long long const& value);
    bool hasNestedContent(std::vector<unsigned long long> const& value);
    bool hasNestedContent(double const& value);
    bool hasNestedContent(std::vector<double> const& value);
    bool hasNestedContent(bool const& value);
    bool hasNestedContent(std::string const& value);
    bool hasNestedContent(std::vector<std::string> const& value);
    bool hasNestedContent(EventID const& value);
    bool hasNestedContent(std::vector<EventID> const& value);
    bool hasNestedContent(LuminosityBlockID const& value);
    bool hasNestedContent(std::vector<LuminosityBlockID> const& value);
    bool hasNestedContent(LuminosityBlockRange const& value);
    bool hasNestedContent(std::vector<LuminosityBlockRange> const& value);
    bool hasNestedContent(EventRange const& value);
    bool hasNestedContent(std::vector<EventRange> const& value);
    bool hasNestedContent(InputTag const& value);
    bool hasNestedContent(std::vector<InputTag> const& value);
    bool hasNestedContent(FileInPath const& value);
  }

  template<typename T>
  class ParameterDescription : public ParameterDescriptionBase {
  public:

    ParameterDescription(std::string const& iLabel,
                         T const& value,
                         bool isTracked,
                         Comment const& iComment = Comment()) :
      // WARNING: the toEnum function is intentionally undefined if the template
      // parameter is ParameterSet or vector<ParameterSet>.  Both of these cases
      // are handled by full template specializations below.  In the first case.
      // ParameterSetDescription should be used instead of ParameterSet.
      // In the second case the function arguments are completely different.
      // Note that this template parameter is most often passed through from
      // an add*<T> function of class ParameterSetDescription. For vector<ParameterSet>
      // use the addVPSet* versions of those functions.
      ParameterDescriptionBase(iLabel, ParameterTypeToEnum::toEnum<T>(), isTracked, true, iComment),
      value_(value) {
    }

    ParameterDescription(char const* iLabel,
                         T const& value,
                         bool isTracked,
                         Comment const& iComment = Comment()) :
      // WARNING: the toEnum function is intentionally undefined if the template
      // parameter is ParameterSet or vector<ParameterSet>.  Both of these cases
      // are handled by full template specializations below.  In the first case.
      // ParameterSetDescription should be used instead of ParameterSet.
      // In the second case the function arguments are completely different.
      // Note that this template parameter is most often passed through from
      // an add*<T> function of class ParameterSetDescription. For vector<ParameterSet>
      // use the addVPSet* versions of those functions.
      ParameterDescriptionBase(iLabel, ParameterTypeToEnum::toEnum<T>(), isTracked, true, iComment),
      value_(value) {
    }

    ParameterDescription(std::string const& iLabel,
                         bool isTracked,
                         Comment const& iComment = Comment()) :
      // WARNING: the toEnum function is intentionally undefined if the template
      // parameter is ParameterSet or vector<ParameterSet>.  Both of these cases
      // are handled by full template specializations below.  In the first case.
      // ParameterSetDescription should be used instead of ParameterSet.
      // In the second case the function arguments are completely different.
      // Note that this template parameter is most often passed through from
      // an add*<T> function of class ParameterSetDescription. For vector<ParameterSet>
      // use the addVPSet* versions of those functions.
      ParameterDescriptionBase(iLabel, ParameterTypeToEnum::toEnum<T>(), isTracked, false, iComment),
      value_() {
    }

    ParameterDescription(char const* iLabel,
                         bool isTracked,
                         Comment const& iComment = Comment()) :
      // WARNING: the toEnum function is intentionally undefined if the template
      // parameter is ParameterSet or vector<ParameterSet>.  Both of these cases
      // are handled by full template specializations below.  In the first case.
      // ParameterSetDescription should be used instead of ParameterSet.
      // In the second case the function arguments are completely different.
      // Note that this template parameter is most often passed through from
      // an add*<T> function of class ParameterSetDescription. For vector<ParameterSet>
      // use the addVPSet* versions of those functions.
      ParameterDescriptionBase(iLabel, ParameterTypeToEnum::toEnum<T>(), isTracked, false, iComment),
      value_() {
    }

    ~ParameterDescription() override { }

    ParameterDescriptionNode* clone() const override {
      return new ParameterDescription(*this);
    }

    T getDefaultValue() const { return value_; }

  private:

    bool exists_(ParameterSet const& pset) const override {
      return pset.existsAs<T>(label(), isTracked());
    }

    bool hasNestedContent_() const override {
      if (!hasDefault()) return false;
      return writeParameterValue::hasNestedContent(value_);
    }

    void writeCfi_(std::ostream& os, int indentation) const override {
      writeParameterValue::writeValue(os, indentation, value_, writeParameterValue::CFI);
    }

    void writeDoc_(std::ostream& os, int indentation) const override {
      writeParameterValue::writeValue(os, indentation, value_, writeParameterValue::DOC);
    }

    bool exists_(ParameterSet const& pset, bool isTracked) const override {
      return pset.existsAs<T>(label(), isTracked);
    }

    void insertDefault_(ParameterSet& pset) const override {
      if (isTracked()) {
        pset.addParameter(label(), value_);
      }
      else {
        pset.addUntrackedParameter(label(), value_);
      }
    }

    T value_;
  };

  template<>
  class ParameterDescription<ParameterSetDescription> : public ParameterDescriptionBase {

  public:

    ParameterDescription(std::string const& iLabel,
                         ParameterSetDescription const& value,
                         bool isTracked,
                         Comment const& iComment = Comment());

    ParameterDescription(char const* iLabel,
                         ParameterSetDescription const& value,
                         bool isTracked,
                         Comment const& iComment = Comment());

    ~ParameterDescription() override;

    ParameterSetDescription const* parameterSetDescription() const override;
    ParameterSetDescription * parameterSetDescription() override;

    ParameterDescriptionNode* clone() const override {
      return new ParameterDescription(*this);
    }

  private:

    void validate_(ParameterSet& pset,
                   std::set<std::string>& validatedLabels,
                   bool optional) const override;

    void printDefault_(std::ostream& os,
                       bool writeToCfi,
                       DocFormatHelper& dfh) const override;

    bool hasNestedContent_() const override;

    void printNestedContent_(std::ostream& os,
                             bool optional,
                             DocFormatHelper& dfh) const override;

    bool exists_(ParameterSet const& pset) const override;

    void writeCfi_(std::ostream& os, int indentation) const override;

    void writeDoc_(std::ostream& os, int indentation) const override;

    bool exists_(ParameterSet const& pset, bool isTracked) const override;

    void insertDefault_(ParameterSet& pset) const override;

    value_ptr<ParameterSetDescription> psetDesc_;
  };

  template<>
  class ParameterDescription<std::vector<ParameterSet> > : public ParameterDescriptionBase {

  public:

    ParameterDescription(std::string const& iLabel,
                         ParameterSetDescription const& psetDesc,
                         bool isTracked,
                         std::vector<ParameterSet> const& vPset,
                         Comment const& iComment = Comment());

    ParameterDescription(char const* iLabel,
                         ParameterSetDescription const& psetDesc,
                         bool isTracked,
                         std::vector<ParameterSet> const& vPset,
                         Comment const& iComment = Comment());

    ParameterDescription(std::string const& iLabel,
                         ParameterSetDescription const& psetDesc,
                         bool isTracked,
                         Comment const& iComment = Comment());

    ParameterDescription(char const* iLabel,
                         ParameterSetDescription const& psetDesc,
                         bool isTracked,
                         Comment const& iComment = Comment());

    ~ParameterDescription() override;

    ParameterSetDescription const* parameterSetDescription() const override;
    ParameterSetDescription * parameterSetDescription() override;

    ParameterDescriptionNode* clone() const override {
      return new ParameterDescription(*this);
    }

    void setPartOfDefaultOfVPSet(bool value) { partOfDefaultOfVPSet_ = value; }

  private:

    void validate_(ParameterSet& pset,
                   std::set<std::string>& validatedLabels,
                   bool optional) const override;

    void printDefault_(std::ostream& os,
                       bool writeToCfi,
                       DocFormatHelper& dfh) const override;

    bool hasNestedContent_() const override;

    void printNestedContent_(std::ostream& os,
                             bool optional,
                             DocFormatHelper& dfh) const override;

    bool exists_(ParameterSet const& pset) const override;

    void writeCfi_(std::ostream& os, int indentation) const override;

    void writeDoc_(std::ostream& os, int indentation) const override;

    bool exists_(ParameterSet const& pset, bool isTracked) const override;

    void insertDefault_(ParameterSet& pset) const override;

    static void writeOneElementToCfi(ParameterSet const& pset,
                                     std::ostream& os,
                                     int indentation,
                                     bool& nextOneStartsWithAComma);

    value_ptr<ParameterSetDescription> psetDesc_;
    std::vector<ParameterSet> vPset_;
    bool partOfDefaultOfVPSet_;
  };
}
#endif
