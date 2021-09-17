#ifndef DDI_ExtrudedPolygon_h
#define DDI_ExtrudedPolygon_h

#include <iosfwd>
#include <vector>

#include "Solid.h"

namespace DDI {

  class ExtrudedPolygon : public Solid {
  public:
    /* G4ExtrudedSolid(const G4String& pName,            */
    /*                 std::vector<G4TwoVector> polygon, */
    /*                 std::vector<ZSection> zsections)  */
    ExtrudedPolygon(const std::vector<double>& x,
                    const std::vector<double>& y,
                    const std::vector<double>& z,
                    const std::vector<double>& zx,
                    const std::vector<double>& zy,
                    const std::vector<double>& zscale);

    double volume() const override;
    void stream(std::ostream&) const override;
  };
}  // namespace DDI
#endif  // DDI_ExtrudedPolygon_h
