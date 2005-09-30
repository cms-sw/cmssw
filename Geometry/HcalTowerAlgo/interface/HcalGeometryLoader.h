#ifndef HcalGeometryLoader_h
#define HcalGeometryLoader_h

#include "Geometry/CaloGeometry/interface/CaloVGeometryLoader.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"

namespace cms {
  class CaloCellGeometry;
  class HcalDetId;

  class HcalGeometryLoader : public CaloVGeometryLoader
  {
  public:
    HcalGeometryLoader();

    virtual void fill(std::vector<DetId> & detIds, std::map<DetId, const CaloCellGeometry*> geometries);

  private:
    /// helper functions to make all the ids and cells, and put them into the
    /// vectors and mpas passed in.
    void fill(HcalSubdetector subdet, int firstEtaRing, int lastEtaRing,
         std::vector<DetId> & detIds, std::map<DetId, const CaloCellGeometry*> geometries);

    CaloCellGeometry * makeCell(const HcalDetId & detId) const;

    HcalTopology theTopology;
    double theHBHEEtaBounds[30];
    double theHFEtaBounds[14];

    double theBarrelDepth;
    double theOuterDepth;
    double theHEDepth[3];
    double theHFDepth[2];

    double theHBThickness;
    double theHFThickness;
    double theHOThickness;
  };

}

#endif

