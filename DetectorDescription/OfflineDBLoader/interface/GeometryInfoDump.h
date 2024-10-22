#include <DetectorDescription/Core/interface/DDCompactView.h>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "DetectorDescription/Core/interface/DDsvalues.h"

class DDCompactView;
class DDPartSelection;

class GeometryInfoDump {
public:
  GeometryInfoDump();
  ~GeometryInfoDump();

  void dumpInfo(bool dumpHistory,
                bool dumpSpecs,
                bool dumpPosInfo,
                const DDCompactView& cpv,
                std::string fname = "GeoHistory",
                int nVols = 0);

private:
  void dumpSpec(const std::vector<std::pair<const DDPartSelection*, const DDsvalues_type*> >& attspec,
                std::ostream& dump);
};
