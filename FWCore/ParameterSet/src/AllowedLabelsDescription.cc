
#include "FWCore/ParameterSet/interface/AllowedLabelsDescription.h"
#include "FWCore/ParameterSet/interface/VParameterSetEntry.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "boost/bind.hpp"

#include <cassert>

namespace edm {

  AllowedLabelsDescription<ParameterSetDescription>::
  AllowedLabelsDescription(std::string const& label,
                           bool isTracked) :
    AllowedLabelsDescriptionBase(label, isTracked),
    psetDesc_()
  {              
  }

  AllowedLabelsDescription<ParameterSetDescription>::
  AllowedLabelsDescription(char const* label,
                           bool isTracked) :
    AllowedLabelsDescriptionBase(label, isTracked),
    psetDesc_()
  {
  }

  AllowedLabelsDescription<ParameterSetDescription>::
  AllowedLabelsDescription(std::string const& label,
                           ParameterSetDescription const& value,
                           bool isTracked) :
    AllowedLabelsDescriptionBase(label, isTracked),
    psetDesc_(new ParameterSetDescription(value))
  {              
  }

  AllowedLabelsDescription<ParameterSetDescription>::
  AllowedLabelsDescription(char const* label,
                           ParameterSetDescription const& value,
                           bool isTracked) :
    AllowedLabelsDescriptionBase(label, isTracked),
    psetDesc_(new ParameterSetDescription(value))
  {
  }

  ParameterDescriptionNode*
  AllowedLabelsDescription<ParameterSetDescription>::
  clone() const {
    return new AllowedLabelsDescription(*this);
  }


  void
  AllowedLabelsDescription<ParameterSetDescription>::
  validateAllowedLabel_(std::string const& allowedLabel,
                        ParameterSet & pset,
                        std::set<std::string> & validatedLabels) const {
    if (pset.existsAs<ParameterSet>(allowedLabel, isTracked())) {
      validatedLabels.insert(allowedLabel);
      if (psetDesc_) {
        ParameterSet * containedPSet = pset.getPSetForUpdate(allowedLabel);
        psetDesc_->validate(*containedPSet);
      }
    }
  }

// -----------------------------------------------------------------------

  AllowedLabelsDescription<std::vector<ParameterSetDescription> >::
  AllowedLabelsDescription(std::string const& label,
                           bool isTracked) :
    AllowedLabelsDescriptionBase(label, isTracked),
    vPsetDesc_()
  {              
  }

  AllowedLabelsDescription<std::vector<ParameterSetDescription> >::
  AllowedLabelsDescription(char const* label,
                           bool isTracked) :
    AllowedLabelsDescriptionBase(label, isTracked),
    vPsetDesc_()
  {
  }

  AllowedLabelsDescription<std::vector<ParameterSetDescription> >::
  AllowedLabelsDescription(std::string const& label,
                           std::vector<ParameterSetDescription> const& value,
                           bool isTracked) :
    AllowedLabelsDescriptionBase(label, isTracked),
    vPsetDesc_(new std::vector<ParameterSetDescription>(value))
  {
  }

  AllowedLabelsDescription<std::vector<ParameterSetDescription> >::
  AllowedLabelsDescription(char const* label,
                           std::vector<ParameterSetDescription> const& value,
                           bool isTracked) :
    AllowedLabelsDescriptionBase(label, isTracked),
    vPsetDesc_(new std::vector<ParameterSetDescription>(value))
  {
  }

  ParameterDescriptionNode*
  AllowedLabelsDescription<std::vector<ParameterSetDescription> >::
  clone() const {
    return new AllowedLabelsDescription(*this);
  }


  void
  AllowedLabelsDescription<std::vector<ParameterSetDescription> >::
  validateAllowedLabel_(std::string const& allowedLabel,
                        ParameterSet & pset,
                        std::set<std::string> & validatedLabels) const {
    if (pset.existsAs<std::vector<ParameterSet> >(allowedLabel, isTracked())) {
      validatedLabels.insert(allowedLabel);

      if (vPsetDesc_) {
        VParameterSetEntry * vpsetEntry = pset.getPSetVectorForUpdate(allowedLabel);
        assert(vpsetEntry);
        if (vpsetEntry->size() != vPsetDesc_->size()) {
          throw edm::Exception(errors::Configuration)
            << "Unexpected number of ParameterSets in vector of parameter sets named \"" << allowedLabel << "\".";
        }
        int i = 0;
        for_all(*vPsetDesc_,
                boost::bind(&AllowedLabelsDescription<std::vector<ParameterSetDescription> >::validateDescription,
                            boost::cref(this),
                            _1,
                            vpsetEntry,
                            boost::ref(i)));
      }
    }
  }

  void
  AllowedLabelsDescription<std::vector<ParameterSetDescription> >::
  validateDescription(ParameterSetDescription const& psetDescription,
                      VParameterSetEntry * vpsetEntry,
                      int & i) const {
    psetDescription.validate(vpsetEntry->psetInVector(i));
    ++i;
  }
}
