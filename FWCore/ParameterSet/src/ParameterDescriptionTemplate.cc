
#include "FWCore/ParameterSet/interface/ParameterDescriptionTemplate.h"

#include "boost/bind.hpp"

namespace edm {

  // =================================================================

  ParameterDescriptionTemplate<ParameterSetDescription>::
  ParameterDescriptionTemplate(std::string const& iLabel,
                               bool isTracked,
                               bool isOptional,
			       ParameterSetDescription const& value) :
      ParameterDescription(iLabel, k_PSet, isTracked, isOptional),
      psetDesc_(value) {
  }

  ParameterDescriptionTemplate<ParameterSetDescription>::
  ParameterDescriptionTemplate(char const* iLabel,
                               bool isTracked,
                               bool isOptional,
			       ParameterSetDescription const& value) :
      ParameterDescription(iLabel, k_PSet, isTracked, isOptional),
      psetDesc_(value) {
  }

  ParameterDescriptionTemplate<ParameterSetDescription>::
  ~ParameterDescriptionTemplate() { }

  void
  ParameterDescriptionTemplate<ParameterSetDescription>::
  validate(ParameterSet const& pset) const {

    bool exists = pset.existsAs<ParameterSet>(label(), isTracked());

    if (!isOptional() && !exists) throwParameterNotDefined();

    if (exists) {
      ParameterSet containedPSet;
      if (isTracked()) {
        containedPSet = pset.getParameter<ParameterSet>(label());
      }
      else {
        containedPSet = pset.getUntrackedParameter<ParameterSet>(label());
      }
      psetDesc_.validate(containedPSet);
    }
  }

  ParameterSetDescription const*
  ParameterDescriptionTemplate<ParameterSetDescription>::
  parameterSetDescription() const {
    return &psetDesc_;
  }

  ParameterSetDescription *
  ParameterDescriptionTemplate<ParameterSetDescription>::
  parameterSetDescription() {
    return &psetDesc_;
  }

  // =================================================================

  ParameterDescriptionTemplate<std::vector<ParameterSetDescription> >::
  ParameterDescriptionTemplate(std::string const& iLabel,
                               bool isTracked,
                               bool isOptional,
			       std::vector<ParameterSetDescription> const& value) :
      ParameterDescription(iLabel, k_VPSet, isTracked, isOptional),
      vPsetDesc_(value) {
  }

  ParameterDescriptionTemplate<std::vector<ParameterSetDescription> >::
  ParameterDescriptionTemplate(char const* iLabel,
                               bool isTracked,
                               bool isOptional,
			       std::vector<ParameterSetDescription> const& value) :
      ParameterDescription(iLabel, k_VPSet, isTracked, isOptional),
      vPsetDesc_(value) {
  }

  ParameterDescriptionTemplate<std::vector<ParameterSetDescription> >::
  ~ParameterDescriptionTemplate() { }

  void
  ParameterDescriptionTemplate<std::vector<ParameterSetDescription> >::
  validate(ParameterSet const& pset) const {

    bool exists = pset.existsAs<std::vector<ParameterSet> >(label(), isTracked());

    if (!isOptional() && !exists) throwParameterNotDefined();

    if (exists) {
      std::vector<ParameterSet> containedPSets;
      if (isTracked()) {
        containedPSets = pset.getParameter<std::vector<ParameterSet> >(label());
      }
      else {
        containedPSets = pset.getUntrackedParameter<std::vector<ParameterSet> >(label());
      }
      if (containedPSets.size() != vPsetDesc_.size()) {
        throw edm::Exception(errors::Configuration)
          << "Unexpected number of ParameterSets in vector of parameter sets named \"" << label() << "\".";
      }
      int i = 0;
      for_all(vPsetDesc_,
              boost::bind(&ParameterDescriptionTemplate<std::vector<ParameterSetDescription> >::validateDescription,
                          boost::cref(this),
                          _1,
                          boost::cref(containedPSets[i]),
                          boost::ref(i)));
    }
  }

  void
  ParameterDescriptionTemplate<std::vector<ParameterSetDescription> >::
  validateDescription(ParameterSetDescription const& psetDescription,
                      ParameterSet const& pset,
                      int & i) const {
    psetDescription.validate(pset);
    ++i;
  }

  std::vector<ParameterSetDescription> const*
  ParameterDescriptionTemplate<std::vector<ParameterSetDescription> >::
  parameterSetDescriptions() const {
    return &vPsetDesc_;
  }

  std::vector<ParameterSetDescription> *
  ParameterDescriptionTemplate<std::vector<ParameterSetDescription> >::
  parameterSetDescriptions() {
    return &vPsetDesc_;
  }

  // =================================================================
}
