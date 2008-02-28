#ifndef Fireworks_Core_BuilderUtils_h
#define Fireworks_Core_BuilderUtils_h

#include <vector>
#include <string>
#include "Rtypes.h"

namespace reco {
   class Track;
}
   
class TEveTrack;
class TEveGeoShapeExtract;
class TGeoBBox;
class TEveElement;
  
namespace fw {
   std::pair<double,double> getPhiRange( const std::vector<double>& phis,
					 double phi );
   TEveTrack* getEveTrack( const reco::Track& track,
			   double max_r = 120,
			   double max_z = 300,
			   double magnetic_field = 4 );
   
   TEveGeoShapeExtract* getShapeExtract( const char* name,
					 TGeoBBox* shape,
					 Color_t color );
   
   void addRhoZEnergyProjection( TEveElement* container,
				 double r_ecal, double z_ecal, 
				 double theta_min, double theta_max, 
				 double phi,
				 Color_t color);
   class NamedCounter
     {
	std::string m_name;
	unsigned int m_index;
      public:
	NamedCounter( std::string name ): 
	  m_name( name ), m_index(0){}
	void operator++() { ++m_index; }
	std::string str() const;
     };
   
}

#endif
