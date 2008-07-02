#ifndef Fireworks_Core_BuilderUtils_h
#define Fireworks_Core_BuilderUtils_h

#include <vector>
#include <string>
#include "Rtypes.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace reco {
   class Track;
   class Candidate;
}
   
class TEveTrack;
class TEveGeoShapeExtract;
class TGeoBBox;
class TEveElement;
class TEveElementList;
class TEveStraightLineSegment;
class TEveStraightLineSet;
class DetId;
class DetIdToMatrix;
  
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
	unsigned int index() const { return m_index; }
     };

     TEveElementList *getEcalCrystals (const EcalRecHitCollection *,
				       const DetIdToMatrix &,
				       const std::vector<DetId> &);
     TEveElementList *getEcalCrystals (const EcalRecHitCollection *,
				       const DetIdToMatrix &,
				       double eta, double phi,
				       int n_eta = 5, int n_phi = 10);
     //    TEveElementList *getMuonCalTowers (double eta, double phi);

     void addStraightLineSegment( TEveStraightLineSet * marker,
				  reco::Candidate const * cand,
				  double scale_factor = 2);
     
}

#endif
