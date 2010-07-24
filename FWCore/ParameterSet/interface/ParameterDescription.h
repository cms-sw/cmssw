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

  class MinimalEventID;
  class LuminosityBlockID;
  class LuminosityBlockRange;
  class EventRange;
  class InputTag;
  class FileInPath;
  class DocFormatHelper;

  namespace writeParameterValue {

    enum ValueFormat { CFI, DOC };

    void writeValue(std::ostream & os, int indentation, int const& value_, ValueFormat format);
    void writeValue(std::ostream & os, int indentation, std::vector<int> const& value_, ValueFormat format);
    void writeValue(std::ostream & os, int indentation, unsigned const& value_, ValueFormat format);
    void writeValue(std::ostream & os, int indentation, std::vector<unsigned> const& value_, ValueFormat format);
    void writeValue(std::ostream & os, int indentation, long long const& value_, ValueFormat format);
    void writeValue(std::ostream & os, int indentation, std::vector<long long> const& value_, ValueFormat format);
    void writeValue(std::ostream & os, int indentation, unsigned long long const& value_, ValueFormat format);
    void writeValue(std::ostream & os, int indentation, std::vector<unsigned long long> const& value_, ValueFormat format);
    void writeValue(std::ostream & os, int indentation, double const& value_, ValueFormat format);
    void writeValue(std::ostream & os, int indentation, std::vector<double> const& value_, ValueFormat format);
    void writeValue(std::ostream & os, int indentation, bool const& value_, ValueFormat format);
    void writeValue(std::ostream & os, int indentation, std::string const& value_, ValueFormat format);
    void writeValue(std::ostream & os, int indentation, std::vector<std::string> const& value_, ValueFormat format);
    void writeValue(std::ostream & os, int indentation, edm::MinimalEventID const& value_, ValueFormat format);
    void writeValue(std::ostream & os, int indentation, std::vector<edm::MinimalEventID> const& value_, ValueFormat format);
    void writeValue(std::ostream & os, int indentation, edm::LuminosityBlockID const& value_, ValueFormat format);
    void writeValue(std::ostream & os, int indentation, std::vector<edm::LuminosityBlockID> const& value_, ValueFormat format);
    void writeValue(std::ostream & os, int indentation, edm::LuminosityBlockRange const& value_, ValueFormat format);
    void writeValue(std::ostream & os, int indentation, std::vector<edm::LuminosityBlockRange> const& value_, ValueFormat format);
    void writeValue(std::ostream & os, int indentation, edm::EventRange const& value_, ValueFormat format);
    void writeValue(std::ostream & os, int indentation, std::vector<edm::EventRange> const& value_, ValueFormat format);
    void writeValue(std::ostream & os, int indentation, edm::InputTag const& value_, ValueFormat format);
    void writeValue(std::ostream & os, int indentation, std::vector<edm::InputTag> const& value_, ValueFormat format);
    void writeValue(std::ostream & os, int indentation, edm::FileInPath const& value_, ValueFormat format);

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
    bool hasNestedContent(edm::MinimalEventID const& value);
    bool hasNestedContent(std::vector<edm::MinimalEventID> const& value);
    bool hasNestedContent(edm::LuminosityBlockID const& value);
    bool hasNestedContent(std::vector<edm::LuminosityBlockID> const& value);
    bool hasNestedContent(edm::LuminosityBlockRange const& value);
    bool hasNestedContent(std::vector<edm::LuminosityBlockRange> const& value);
    bool hasNestedContent(edm::EventRange const& value);
    bool hasNestedContent(std::vector<edm::EventRange> const& value);
    bool hasNestedContent(edm::InputTag const& value);
    bool hasNestedContent(std::vector<edm::InputTag> const& value);
    bool hasNestedContent(edm::FileInPath const& value);
  }

  template<class T>
  class ParameterDescription : public ParameterDescriptionBase {
  public:

    ParameterDescription(std::string const& iLabel,
                         T const& value,
                         bool isTracked
                        ):
      // WARNING: the toEnum function is intentionally undefined if the template
      // parameter is ParameterSet or vector<ParameterSet>.  Both of these cases
      // are handled by full template specializations below.  In the first case.
      // ParameterSetDescription should be used instead of ParameterSet.
      // In the second case the function arguments are completely different.
      // Note that this template parameter is most often passed through from
      // an add*<T> function of class ParameterSetDescription. For vector<ParameterSet>
      // use the addVPSet* versions of those functions.
      ParameterDescriptionBase(iLabel, ParameterTypeToEnum::toEnum<T>(), isTracked, true),
      value_(value) {
    }

    ParameterDescription(char const* iLabel,
                         T const& value,
                         bool isTracked
                        ):
      // WARNING: the toEnum function is intentionally undefined if the template
      // parameter is ParameterSet or vector<ParameterSet>.  Both of these cases
      // are handled by full template specializations below.  In the first case.
      // ParameterSetDescription should be used instead of ParameterSet.
      // In the second case the function arguments are completely different.
      // Note that this template parameter is most often passed through from
      // an add*<T> function of class ParameterSetDescription. For vector<ParameterSet>
      // use the addVPSet* versions of those functions.
      ParameterDescriptionBase(iLabel, ParameterTypeToEnum::toEnum<T>(), isTracked, true),
      value_(value) {
    }

    ParameterDescription(std::string const& iLabel,
                         bool isTracked
                        ):
      // WARNING: the toEnum function is intentionally undefined if the template
      // parameter is ParameterSet or vector<ParameterSet>.  Both of these cases
      // are handled by full template specializations below.  In the first case.
      // ParameterSetDescription should be used instead of ParameterSet.
      // In the second case the function arguments are completely different.
      // Note that this template parameter is most often passed through from
      // an add*<T> function of class ParameterSetDescription. For vector<ParameterSet>
      // use the addVPSet* versions of those functions.
      ParameterDescriptionBase(iLabel, ParameterTypeToEnum::toEnum<T>(), isTracked, false),
      value_() {
    }

    ParameterDescription(char const* iLabel,
                         bool isTracked
                        ):
      // WARNING: the toEnum function is intentionally undefined if the template
      // parameter is ParameterSet or vector<ParameterSet>.  Both of these cases
      // are handled by full template specializations below.  In the first case.
      // ParameterSetDescription should be used instead of ParameterSet.
      // In the second case the function arguments are completely different.
      // Note that this template parameter is most often passed through from
      // an add*<T> function of class ParameterSetDescription. For vector<ParameterSet>
      // use the addVPSet* versions of those functions.
      ParameterDescriptionBase(iLabel, ParameterTypeToEnum::toEnum<T>(), isTracked, false),
      value_() {
    }

    virtual ~ParameterDescription() { }

    virtual ParameterDescriptionNode* clone() const {
      return new ParameterDescription(*this);
    }

    T getDefaultValue() const { return value_; }

  private:

    virtual bool exists_(ParameterSet const& pset) const {
      return pset.existsAs<T>(label(), isTracked());
    }

    virtual bool hasNestedContent_() {
      if (!hasDefault()) return false;
      return writeParameterValue::hasNestedContent(value_);
    }

    virtual void writeCfi_(std::ostream & os, int indentation) const {
      writeParameterValue::writeValue(os, indentation, value_, writeParameterValue::CFI);
    }

    virtual void writeDoc_(std::ostream & os, int indentation) const {
      writeParameterValue::writeValue(os, indentation, value_, writeParameterValue::DOC);
    }

    virtual bool exists_(ParameterSet const& pset, bool isTracked) const {
      return pset.existsAs<T>(label(), isTracked);
    }

    virtual void insertDefault_(ParameterSet & pset) const {
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
                         bool isTracked
                        );

    ParameterDescription(char const* iLabel,
                         ParameterSetDescription const& value,
                         bool isTracked
                        );

    virtual ~ParameterDescription();

    virtual ParameterSetDescription const* parameterSetDescription() const;
    virtual ParameterSetDescription * parameterSetDescription();

    virtual ParameterDescriptionNode* clone() const {
      return new ParameterDescription(*this);
    }

  private:

    virtual void validate_(ParameterSet & pset,
                           std::set<std::string> & validatedLabels,
                           bool optional) const;

    virtual void printDefault_(std::ostream & os,
                                 bool writeToCfi,
                                 DocFormatHelper & dfh);

    virtual bool hasNestedContent_();

    virtual void printNestedContent_(std::ostream & os,
                                     bool optional,
                                     DocFormatHelper & dfh);

    virtual bool exists_(ParameterSet const& pset) const;

    virtual void writeCfi_(std::ostream & os, int indentation) const;

    virtual void writeDoc_(std::ostream & os, int indentation) const;

    virtual bool exists_(ParameterSet const& pset, bool isTracked) const;

    virtual void insertDefault_(ParameterSet & pset) const;

    value_ptr<ParameterSetDescription> psetDesc_;
  };

  template<>
  class ParameterDescription<std::vector<ParameterSet> > : public ParameterDescriptionBase {

  public:

    ParameterDescription(std::string const& iLabel,
                         ParameterSetDescription const& psetDesc,
                         bool isTracked,
                         std::vector<ParameterSet> const& vPset
                        );

    ParameterDescription(char const* iLabel,
                         ParameterSetDescription const& psetDesc,
                         bool isTracked,
                         std::vector<ParameterSet> const& vPset
                        );

    ParameterDescription(std::string const& iLabel,
                         ParameterSetDescription const& psetDesc,
                         bool isTracked
                        );

    ParameterDescription(char const* iLabel,
                         ParameterSetDescription const& psetDesc,
                         bool isTracked
                        );

    virtual ~ParameterDescription();

    virtual ParameterSetDescription const* parameterSetDescription() const;
    virtual ParameterSetDescription * parameterSetDescription();

    virtual ParameterDescriptionNode* clone() const {
      return new ParameterDescription(*this);
    }

    void setPartOfDefaultOfVPSet(bool value) { partOfDefaultOfVPSet_ = value; }

  private:

    virtual void validate_(ParameterSet & pset,
                           std::set<std::string> & validatedLabels,
                           bool optional) const;

    virtual void printDefault_(std::ostream & os,
                               bool writeToCfi,
                               DocFormatHelper & dfh);

    virtual bool hasNestedContent_();

    virtual void printNestedContent_(std::ostream & os,
                                     bool optional,
                                     DocFormatHelper & dfh);

    virtual bool exists_(ParameterSet const& pset) const;

    virtual void writeCfi_(std::ostream & os, int indentation) const;

    virtual void writeDoc_(std::ostream & os, int indentation) const;

    virtual bool exists_(ParameterSet const& pset, bool isTracked) const;

    virtual void insertDefault_(ParameterSet & pset) const;

    static void writeOneElementToCfi(ParameterSet const& pset,
                                     std::ostream & os,
                                     int indentation,
                                     bool & nextOneStartsWithAComma);

    value_ptr<ParameterSetDescription> psetDesc_;
    std::vector<ParameterSet> vPset_;
    bool partOfDefaultOfVPSet_;
  };
}
#endif
