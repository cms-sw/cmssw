
#include "FWCore/ParameterSet/interface/AllowedLabelsDescription.h"
#include "FWCore/ParameterSet/interface/VParameterSetEntry.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

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
      printSpaces(os, indentation + DocFormatHelper::offsetSectionContent());
    }
    else {
      dfh.indent2(os);
    }
    os << "see Section " << newSection << "\n";
    if (!dfh.brief()) os << "\n";

    printSpaces(os, indentation);
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

  AllowedLabelsDescription<std::vector<ParameterSet> >::
  AllowedLabelsDescription(std::string const& label,
                           bool isTracked) :
    AllowedLabelsDescriptionBase(label, k_VPSet, isTracked),
    psetDesc_()
  {              
  }

  AllowedLabelsDescription<std::vector<ParameterSet> >::
  AllowedLabelsDescription(char const* label,
                           bool isTracked) :
    AllowedLabelsDescriptionBase(label, k_VPSet, isTracked),
    psetDesc_()
  {
  }

  AllowedLabelsDescription<std::vector<ParameterSet> >::
  AllowedLabelsDescription(std::string const& label,
                           ParameterSetDescription const& value,
                           bool isTracked) :
    AllowedLabelsDescriptionBase(label, k_VPSet, isTracked),
    psetDesc_(new ParameterSetDescription(value))
  {
  }

  AllowedLabelsDescription<std::vector<ParameterSet> >::
  AllowedLabelsDescription(char const* label,
                           ParameterSetDescription const& value,
                           bool isTracked) :
    AllowedLabelsDescriptionBase(label, k_VPSet, isTracked),
    psetDesc_(new ParameterSetDescription(value))
  {
  }

  ParameterDescriptionNode*
  AllowedLabelsDescription<std::vector<ParameterSet> >::
  clone() const {
    return new AllowedLabelsDescription(*this);
  }

  void
  AllowedLabelsDescription<std::vector<ParameterSet> >::
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
      printSpaces(os, indentation + DocFormatHelper::offsetSectionContent());
    }
    else {
      dfh.indent2(os);
    }
    os << "see Section " << newSection << "\n";
    if (!dfh.brief()) os << "\n";

    printSpaces(os, indentation);
    os << "Section " << newSection
       << " PSet description used to validate all elements of VPSet's:\n";
    if (!dfh.brief()) os << "\n";

    DocFormatHelper new_dfh(dfh);
    new_dfh.setSection(newSection);
    new_dfh.setIndentation(indentation + DocFormatHelper::offsetSectionContent());
    new_dfh.setParent(DocFormatHelper::OTHER);

    psetDesc_->print(os, new_dfh);
  }

  void
  AllowedLabelsDescription<std::vector<ParameterSet> >::
  validateAllowedLabel_(std::string const& allowedLabel,
                        ParameterSet & pset,
                        std::set<std::string> & validatedLabels) const {
    if (pset.existsAs<std::vector<ParameterSet> >(allowedLabel, isTracked())) {
      validatedLabels.insert(allowedLabel);

      if (psetDesc_) {
        VParameterSetEntry * vpsetEntry = pset.getPSetVectorForUpdate(allowedLabel);
        assert(vpsetEntry);
        for (unsigned i = 0; i < vpsetEntry->size(); ++i) {
          psetDesc_->validate(vpsetEntry->psetInVector(i));
        }
      }
    }
  }
}
