#ifndef DETECTOR_DESCRIPTION_DD_COMPACT_VIEW_H
#define DETECTOR_DESCRIPTION_DD_COMPACT_VIEW_H

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

#include "DetectorDescription/DDCMS/interface/DDSpecParRegistry.h"
#include "DetectorDescription/DDCMS/interface/DDDetector.h"

namespace cms {

  class DDCompactView {
  public:
    DDCompactView(const cms::DDDetector& det) : m_det(det) {}
    const cms::DDDetector* detector() const { return &m_det; }
    DDSpecParRegistry const& specpars() const { return m_det.specpars(); }
    template <typename T>
    std::vector<T> getVector(const std::string&) const;

  private:
    const cms::DDDetector& m_det;
  };
}  // namespace cms

#endif
