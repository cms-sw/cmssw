#include <DetectorDescription/Core/interface/DDCompactView.h>
#include <iostream>

class GeometryInfoDump {

public:
 
   GeometryInfoDump( );
  ~GeometryInfoDump();

  void dumpInfo ( bool dumpHistory, bool dumpSpecs, bool dumpPosInfo
		, const DDCompactView& cpv, std::string fname = "GeoHistory", int nVols = 0 );

 private:
  void dumpSpec( const std::vector<std::pair< DDPartSelection*, DDsvalues_type*> >& attspec, std::ostream& dump );
};

