#ifndef DetectorDescription_DDCMS_DDCompactView_h
#define DetectorDescription_DDCMS_DDCompactView_h

// -*- C++ -*-
//
// Package:    DetectorDescription/Core
// Class:      DDCompactView
//
/**\class DDCompactView

 Description: DD Compact View Facade

 Implementation:
     The DDCompactView facade serves as a launching point for a broader
     refactor of monolithic or tightly-coupled systems in favor of more
     loosely-coupled code.
*/
//
// Original Author:  Ianna Osborne
//         Created:  Wed, 22 May 2019 12:51:22 GMT
//
//

#include "DetectorDescription/DDCMS/interface/DDDetector.h"
#include <DD4hep/SpecParRegistry.h>

namespace cms {
  using DDSpecParRegistry = dd4hep::SpecParRegistry;
  using DDSpecParRefs = dd4hep::SpecParRefs;

  class DDCompactView {
  public:
    DDCompactView(const cms::DDDetector& det) : m_det(det) {}
    const cms::DDDetector* detector() const { return &m_det; }
    DDSpecParRegistry const& specpars() const { return m_det.specpars(); }
    template <typename T>
    std::vector<T> getVector(const std::string&) const;

    template <typename T>
    T const& get(const std::string&) const;
    template <typename T>
    T const& get(const std::string&, const std::string&) const;

  private:
    const cms::DDDetector& m_det;
  };

  /* Helper: For a given node, get the values associated to a given parameter, from the XMLs SpecPar sections.
 * NB: The same parameter can appear several times WITHIN the same SpecPar section (hence, we have a std::vector).
 * WARNING: This stops at the first relevant SpecPar section encountered.
 * Hence, if A GIVEN NODE HAS SEVERAL SPECPAR XML SECTIONS RE-DEFINING THE SAME PARAMETER,
 * only the first XML SpecPar block will be considered.
 */
  template <typename T>
  std::vector<T> getAllParameterValuesFromSpecParSections(const cms::DDSpecParRegistry& allSpecParSections,
                                                          const std::string& nodePath,
                                                          const std::string& parameterName) {
    cms::DDSpecParRefs filteredSpecParSections;
    allSpecParSections.filter(filteredSpecParSections, parameterName);
    for (const auto& mySpecParSection : filteredSpecParSections) {
      if (mySpecParSection.second->hasPath(nodePath)) {
        return mySpecParSection.second->value<std::vector<T>>(parameterName);
      }
    }

    return std::vector<T>();
  }

  /* Helper: For a given node, get the value associated to a given parameter, from the XMLs SpecPar sections.
 * This is the parameterValueIndex-th value (within a XML SpecPar block.) of the desired parameter. 
 */
  template <typename T>
  T getParameterValueFromSpecParSections(const cms::DDSpecParRegistry& allSpecParSections,
                                         const std::string& nodePath,
                                         const std::string& parameterName,
                                         const unsigned int parameterValueIndex) {
    const std::vector<T>& allParameterValues =
        getAllParameterValuesFromSpecParSections<T>(allSpecParSections, nodePath, parameterName);
    if (parameterValueIndex < allParameterValues.size()) {
      return allParameterValues.at(parameterValueIndex);
    }
    return T();
  }

}  // namespace cms

#endif
