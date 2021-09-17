#ifndef DDD_DDI_CUTTUBS_H
#define DDD_DDI_CUTTUBS_H

#include <iostream>
#include "Solid.h"

namespace DDI {

  class CutTubs : public Solid {
  public:
    CutTubs(double zHalf,
            double rIn,
            double rOut,
            double startPhi,
            double deltaPhi,
            double lx,
            double ly,
            double lz,
            double tx,
            double ty,
            double tz);

    double volume() const override { return -1; }

    void stream(std::ostream& os) const override;
  };
}  // namespace DDI

#endif
