
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
    AllowedLabelsDescriptionBase(label, k_PSet, isTracked),
    psetDesc_()
  {              
  }

  AllowedLabelsDescription<ParameterSetDescription>::
  AllowedLabelsDescription(char const* label,
                           bool isTracked) :
    AllowedLabelsDescriptionBase(label, k_PSet, isTracked),
    psetDesc_()
  {
  }

  AllowedLabelsDescription<ParameterSetDescription>::
  AllowedLabelsDescription(std::string const& label,
                           ParameterSetDescription const& value,
                           bool isTracked) :
    AllowedLabelsDescriptionBase(label, k_PSet, isTracked),
    psetDesc_(new ParameterSetDescription(value))
  {              
  }

  AllowedLabelsDescription<ParameterSetDescription>::
  AllowedLabelsDescription(char const* label,
                           ParameterSetDescription const& value,
                           bool isTracked) :
    AllowedLabelsDescriptionBase(label, k_PSet, isTracked),
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
  printNestedContent_(std::ostream & os,
                      bool optional,
                      DocFormatHelper & dfh) {

    printNestedContentBase_(os, optional, dfh);

    int indentation = dfh.indentation();
    if (dfh.parent() != DocFormatHelper::TOP) {
      indentation -= DocFormatHelper::offsetSectionContent();
    }

    std::stringstream ss;
    ss << dfh.section() << "." << dfh.counter() << ".1";
    std::string newSection = ss.str();

    if (dfh.brief()) {
      os << std::setfill(' ')
         << std::setw(indentation + DocFormatHelper::offsetSectionContent())
         << "";
    }
    else {
      dfh.indent2(os);
    }
    os << "see Section " << newSection << "\n";
    if (!dfh.brief()) os << "\n";

    os << std::setfill(' ') << std::setw(indentation) << "";
    os << "Section " << newSection
       << " PSet description:\n";
    if (!dfh.brief()) os << "\n";

    DocFormatHelper new_dfh(dfh);
    new_dfh.setSection(newSection);
    new_dfh.setIndentation(indentation + DocFormatHelper::offsetSectionContent());
    new_dfh.setParent(DocFormatHelper::OTHER);

    psetDesc_->print(os, new_dfh);
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
    AllowedLabelsDescriptionBase(label, k_VPSet, isTracked),
    vPsetDesc_()
  {              
  }

  AllowedLabelsDescription<std::vector<ParameterSetDescription> >::
  AllowedLabelsDescription(char const* label,
                           bool isTracked) :
    AllowedLabelsDescriptionBase(label, k_VPSet, isTracked),
    vPsetDesc_()
  {
  }

  AllowedLabelsDescription<std::vector<ParameterSetDescription> >::
  AllowedLabelsDescription(std::string const& label,
                           std::vector<ParameterSetDescription> const& value,
                           bool isTracked) :
    AllowedLabelsDescriptionBase(label, k_VPSet, isTracked),
    vPsetDesc_(new std::vector<ParameterSetDescription>(value))
  {
  }

  AllowedLabelsDescription<std::vector<ParameterSetDescription> >::
  AllowedLabelsDescription(char const* label,
                           std::vector<ParameterSetDescription> const& value,
                           bool isTracked) :
    AllowedLabelsDescriptionBase(label, k_VPSet, isTracked),
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
  printNestedContent_(std::ostream & os,
                      bool optional,
                      DocFormatHelper & dfh) {

    printNestedContentBase_(os, optional, dfh);

    int indentation = dfh.indentation();
    if (dfh.parent() != DocFormatHelper::TOP) {
      indentation -= DocFormatHelper::offsetSectionContent();
    }

    std::stringstream ss;
    ss << dfh.section() << "." << dfh.counter() << ".1";
    std::string newSection = ss.str();

    if (dfh.brief()) {
      os << std::setfill(' ')
         << std::setw(indentation + DocFormatHelper::offsetSectionContent())
         << "";
    }
    else {
      dfh.indent2(os);
    }
    os << "see Section " << newSection << "\n";
    if (!dfh.brief()) os << "\n";

    os << std::setfill(' ') << std::setw(indentation) << "";
    os << "Section " << newSection
       << " VPSet description:\n";

    for (unsigned i = 1; i <= vPsetDesc_->size(); ++i) {
      os << std::setfill(' ')
         << std::setw(indentation + DocFormatHelper::offsetSectionContent())
         << "";
      os << "[" << (i - 1) << "]: see Section " << dfh.section() << "." << dfh.counter()
         << ".1." << i << "\n";
    }
    if (!dfh.brief()) os << "\n";

    for (unsigned i = 1; i <= vPsetDesc_->size(); ++i) {

      std::stringstream ss2;
      ss2 << newSection << "." << i;
      std::string newSection2 = ss2.str();

      os << std::setfill(' ') << std::setw(indentation) << "";
      os << "Section " << newSection2 << " PSet description:\n";
      if (!dfh.brief()) os << "\n";

      DocFormatHelper new_dfh(dfh);
      new_dfh.setSection(newSection2);
      new_dfh.setIndentation(indentation + DocFormatHelper::offsetSectionContent());
      new_dfh.setParent(DocFormatHelper::OTHER);
      (*vPsetDesc_)[i - 1].print(os, new_dfh);
    }
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
