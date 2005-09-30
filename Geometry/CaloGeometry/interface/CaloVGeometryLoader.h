#ifndef CaloVGeometryLoader_h
#define CaloVGeometryLoader_h

#include<vector>
#include<map>

namespace cms {
class DetId;
class CaloCellGeometry;

class CaloVGeometryLoader {
public:
  virtual void fill(std::vector<DetId> & detIds, std::map<DetId, const CaloCellGeometry*> geometries) = 0;
};

}

#endif

