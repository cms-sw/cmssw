#ifndef Fireworks_Core_BuilderUtils_h
#define Fireworks_Core_BuilderUtils_h

#include <vector>
#include <string>
#include "Rtypes.h"

class TEveGeoShape;
class TEveElement;
class TGeoBBox;
class FWProxyBuilderBase;

namespace edm {
   class EventBase;
}

namespace fireworks
{
   std::pair<double,double> getPhiRange( const std::vector<double>& phis,
                                         double phi );
   TEveGeoShape* getShape( const char* name,
                           TGeoBBox* shape,
                           Color_t color );

   void addRhoZEnergyProjection( FWProxyBuilderBase*, TEveElement*,
                                 double r_ecal, double z_ecal,
                                 double theta_min, double theta_max,
                                 double phi );

   std::string getTimeGMT( const edm::EventBase& event );
   std::string getLocalTime( const edm::EventBase& event );

   void invertBox( std::vector<float> &corners );
   void addBox( const std::vector<float> &corners, TEveElement*,  FWProxyBuilderBase*);
   void addCircle( double eta, double phi, double radius, const unsigned int nLineSegments, TEveElement* comp, FWProxyBuilderBase* pb );
   void addDashedArrow( double phi, double size, TEveElement* comp, FWProxyBuilderBase* pb );
   void addDashedLine( double phi, double theta, double size, TEveElement* comp, FWProxyBuilderBase* pb );
   void addDoubleLines( double phi, TEveElement* comp, FWProxyBuilderBase* pb );

   //
   //  box-utilts
   // 
   void energyScaledBox3DCorners( const float* corners, float scale, std::vector<float>&, bool invert = false);
   void drawEnergyScaledBox3D   ( const float* corners, float scale, TEveElement*,  FWProxyBuilderBase*, bool invert = false );

   void energyTower3DCorners( const float* corners, float scale, std::vector<float>&, bool reflect = false);
   void drawEnergyTower3D   ( const float* corners, float scale, TEveElement*, FWProxyBuilderBase*, bool reflect = false );
 
   // AMT: is this needed ?
   void etScaledBox3DCorners( const float* corners, float energy, float maxEnergy,  std::vector<float>& scaledCorners, bool reflect = false );
   void drawEtScaledBox3D( const float* corners, float energy, float maxEnergy, TEveElement*,  FWProxyBuilderBase*, bool reflect = false );
  
   void etTower3DCorners( const float* corners, float scale, std::vector<float>&, bool reflect = false);
   void drawEtTower3D( const float* corners, float scale, TEveElement*, FWProxyBuilderBase*, bool reflect = false );
}

#endif // Fireworks_Core_BuilderUtils_h
