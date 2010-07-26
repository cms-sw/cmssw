#ifndef Fireworks_Core_BuilderUtils_h
#define Fireworks_Core_BuilderUtils_h

#include <vector>
#include <string>
#include "Rtypes.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"


class TEveTrack;
class TEveGeoShape;
class TGeoBBox;
class TEveElement;
class TEveElementList;
class TEveStraightLineSet;
class DetId;
class DetIdToMatrix;
class FWProxyBuilderBase;

namespace edm {
   class EventBase;
}

namespace reco {
   class Candidate;
}

namespace fw {
std::pair<double,double> getPhiRange( const std::vector<double>& phis,
                                      double phi );
TEveGeoShape* getShape( const char* name,
                        TGeoBBox* shape,
                        Color_t color );

void addRhoZEnergyProjection( FWProxyBuilderBase*, TEveElement*,
                              double r_ecal, double z_ecal,
                              double theta_min, double theta_max,
                              double phi);
class NamedCounter
{
   std::string m_name;
   unsigned int m_index;
public:
   NamedCounter( std::string name ) :
      m_name( name ), m_index(0){
   }
   void operator++() {
      ++m_index;
   }
   std::string str() const;
   unsigned int index() const {
      return m_index;
   }
};

TEveElementList *getEcalCrystals (const EcalRecHitCollection *,
                                  const DetIdToMatrix &,
                                  const std::vector<DetId> &);
TEveElementList *getEcalCrystals (const EcalRecHitCollection *,
                                  const DetIdToMatrix &,
                                  double eta, double phi,
                                  int n_eta = 5, int n_phi = 10);
//    TEveElementList *getMuonCalTowers (double eta, double phi);

std::string getTimeGMT(const edm::EventBase& event);
std::string getLocalTime(const edm::EventBase& event);
}

#endif
