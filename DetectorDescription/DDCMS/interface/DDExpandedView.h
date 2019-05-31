#ifndef DETECTOR_DESCRIPTION_DD_EXPANDED_VIEW_H
#define DETECTOR_DESCRIPTION_DD_EXPANDED_VIEW_H

// -*- C++ -*-
//
// Package:    DetectorDescription/Core
// Class:      DDExpandedView
//
/**\class DDExpandedView

 Description: DD Expanded View Facade

 Implementation:
     The DDExpandedView facade serves as a launching point for a broader
     refactor of monolithic or tightly-coupled systems in favor of more
     loosely-coupled code.
*/
//
// Original Author:  Ianna Osborne
//         Created:  Wed, 22 May 2019 12:59:09 GMT
//
//

namespace cms {

  class DDCompactView;
  class DDFilteredView;

  class DDExpandedView {
    friend class cms::DDFilteredView;

  public:
    DDExpandedView(const cms::DDCompactView& cpv) : m_cpv(cpv) {}

  private:
    const cms::DDCompactView& m_cpv;
  };
}  // namespace cms

#endif
