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

#include "boost/cstdint.hpp"

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
  class VParameterSetEntry;

  namespace writeToCfi {
    void writeValueToCfi(std::ostream & os, int indentation, int const& value_);
    void writeValueToCfi(std::ostream & os, int indentation, std::vector<int> const& value_);
    void writeValueToCfi(std::ostream & os, int indentation, unsigned const& value_);
    void writeValueToCfi(std::ostream & os, int indentation, std::vector<unsigned> const& value_);
    void writeValueToCfi(std::ostream & os, int indentation, boost::int64_t const& value_);
    void writeValueToCfi(std::ostream & os, int indentation, std::vector<boost::int64_t> const& value_);
    void writeValueToCfi(std::ostream & os, int indentation, boost::uint64_t const& value_);
    void writeValueToCfi(std::ostream & os, int indentation, std::vector<boost::uint64_t> const& value_);
    void writeValueToCfi(std::ostream & os, int indentation, double const& value_);
    void writeValueToCfi(std::ostream & os, int indentation, std::vector<double> const& value_);
    void writeValueToCfi(std::ostream & os, int indentation, bool const& value_);
    void writeValueToCfi(std::ostream & os, int indentation, std::string const& value_);
    void writeValueToCfi(std::ostream & os, int indentation, std::vector<std::string> const& value_);
    void writeValueToCfi(std::ostream & os, int indentation, edm::EventID const& value_);
    void writeValueToCfi(std::ostream & os, int indentation, std::vector<edm::EventID> const& value_);
    void writeValueToCfi(std::ostream & os, int indentation, edm::LuminosityBlockID const& value_);
    void writeValueToCfi(std::ostream & os, int indentation, std::vector<edm::LuminosityBlockID> const& value_);
    void writeValueToCfi(std::ostream & os, int indentation, edm::LuminosityBlockRange const& value_);
    void writeValueToCfi(std::ostream & os, int indentation, std::vector<edm::LuminosityBlockRange> const& value_);
    void writeValueToCfi(std::ostream & os, int indentation, edm::EventRange const& value_);
    void writeValueToCfi(std::ostream & os, int indentation, std::vector<edm::EventRange> const& value_);
    void writeValueToCfi(std::ostream & os, int indentation, edm::InputTag const& value_);
    void writeValueToCfi(std::ostream & os, int indentation, std::vector<edm::InputTag> const& value_);
    void writeValueToCfi(std::ostream & os, int indentation, edm::FileInPath const& value_);
  }

  template<class T>
  class ParameterDescription : public ParameterDescriptionBase {
  public:

    ParameterDescription(std::string const& iLabel,
                         T const& value,
                         bool isTracked
                        ):
      // WARNING: the toEnum function is intentionally undefined if the template
      // parameter is ParameterSet or vector<ParameterSet>.  ParameterSetDescription
      // or vector<ParameterSetDescription> should be used instead.  This template
      // parameter is usually passed through from an add*<T> function of ParameterSetDescription.
      ParameterDescriptionBase(iLabel, ParameterTypeToEnum::toEnum<T>(), isTracked),
      value_(value) {
    }

    ParameterDescription(char const* iLabel,
                         T const& value,
                         bool isTracked
                        ):
      // WARNING: the toEnum function is intentionally undefined if the template
      // parameter is ParameterSet or vector<ParameterSet>.  ParameterSetDescription
      // or vector<ParameterSetDescription> should be used instead.  This template
      // parameter is usually passed through from an add*<T> function of ParameterSetDescription.
      ParameterDescriptionBase(iLabel, ParameterTypeToEnum::toEnum<T>(), isTracked),
      value_(value) {
    }

    virtual ~ParameterDescription() { }

    virtual ParameterDescriptionNode* clone() const {
      return new ParameterDescription(*this);
    }

    T getDefaultValue() const { return value_; }

  private:

    virtual void validate_(ParameterSet & pset,
                           std::set<std::string> & validatedLabels,
                           bool optional) const {

      bool exists = pset.existsAs<T>(label(), isTracked());

      if (exists) {
        validatedLabels.insert(label());
      }
      else if (pset.existsAs<T>(label(), !isTracked())) {
        throwParameterWrongTrackiness();
      }
      else if (pset.exists(label())) {
        throwParameterWrongType();
      }

      if (!optional && !exists) {
        if (isTracked()) {
          pset.addParameter(label(), value_);
        }
        else {
          pset.addUntrackedParameter(label(), value_);
        }
        validatedLabels.insert(label());
      }
    }

    virtual bool exists_(ParameterSet const& pset) const {
      return pset.existsAs<T>(label(), isTracked());
    }

    virtual void writeCfi_(std::ostream & os, int indentation) const {
      writeToCfi::writeValueToCfi(os, indentation, value_);
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

    virtual bool exists_(ParameterSet const& pset) const;

    virtual void writeCfi_(std::ostream & os, int indentation) const;

    value_ptr<ParameterSetDescription> psetDesc_;
  };

  template<>
  class ParameterDescription<std::vector<ParameterSetDescription> > : public ParameterDescriptionBase {

  public:

    ParameterDescription(std::string const& iLabel,
                         std::vector<ParameterSetDescription> const& vPsetDesc,
                         bool isTracked
                        );

    ParameterDescription(char const* iLabel,
                         std::vector<ParameterSetDescription> const& vPsetDesc,
                         bool isTracked
                        );

    virtual ~ParameterDescription();

    virtual std::vector<ParameterSetDescription> const* parameterSetDescriptions() const;
    virtual std::vector<ParameterSetDescription> * parameterSetDescriptions();

    virtual ParameterDescriptionNode* clone() const {
      return new ParameterDescription(*this);
    }

  private:

    virtual void validate_(ParameterSet & pset,
                           std::set<std::string> & validatedLabels,
                           bool optional) const;

    virtual bool exists_(ParameterSet const& pset) const;

    virtual void writeCfi_(std::ostream & os, int indentation) const;

    static void writeOneDescriptionToCfi(ParameterSetDescription const& psetDesc,
                                         std::ostream & os,
                                         int indentation,
                                         bool & nextOneStartsWithAComma);
    void
    validateDescription(ParameterSetDescription const& psetDescription,
                        VParameterSetEntry * vpsetEntry,
                        int & i) const;

    value_ptr<std::vector<ParameterSetDescription> > vPsetDesc_;
  };
}
#endif
