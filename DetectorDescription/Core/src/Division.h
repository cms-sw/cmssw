#ifndef DETECTOR_DESCRIPTION_CORE_DDI_DIVISION_H
#define DETECTOR_DESCRIPTION_CORE_DDI_DIVISION_H

#include <iostream>
#include <map>
#include <utility>
#include <vector>

#include "DetectorDescription/Core/interface/DDAxes.h"
#include "DetectorDescription/Core/interface/DDLogicalPart.h"

namespace DDI {
  class Division {
  public:
    Division(const DDLogicalPart &parent, DDAxes axis, int nReplicas, double width, double offset);

    // Constructor with number of divisions
    Division(const DDLogicalPart &parent, DDAxes axis, int nReplicas, double offset);

    // Constructor with width
    Division(const DDLogicalPart &parent, DDAxes axis, double width, double offset);

    DDAxes axis() const;
    int nReplicas() const;
    double width() const;
    double offset() const;
    const DDLogicalPart &parent() const;
    void stream(std::ostream &);

  private:
    DDLogicalPart parent_;
    DDAxes axis_;
    int nReplicas_;
    double width_;
    double offset_;
  };
}  // namespace DDI

#endif
