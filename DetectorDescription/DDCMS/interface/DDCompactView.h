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

namespace cms {

  class DDDetector;

  class DDCompactView {
  public:
    DDCompactView(const cms::DDDetector& det) : m_det(det) {}

  private:
    const cms::DDDetector& m_det;
  };
}  // namespace cms

#endif
