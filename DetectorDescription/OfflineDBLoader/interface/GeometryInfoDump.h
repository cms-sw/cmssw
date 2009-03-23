#include <DetectorDescription/Core/interface/DDCompactView.h>

class GeometryInfoDump {

public:
 
   GeometryInfoDump( );
  ~GeometryInfoDump();

  void dumpInfo ( bool dumpHistory, bool dumpSpecs, bool dumpPosInfo
		, const DDCompactView& cpv, std::string fname = "GeoHistory", int nVols = 0 );

};

