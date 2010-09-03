#ifndef Fireworks_Calo_CaloUtils_h
#define Fireworks_Calo_CaloUtils_h

#include <vector>

class TEveElement;
class FWProxyBuilderBase;

namespace fireworks
{
   void invertBox( std::vector<float> &corners );
   void addBox( const std::vector<float> &corners, TEveElement*,  FWProxyBuilderBase*);
   void addCircle( double eta, double phi, double radius, const unsigned int nLineSegments, TEveElement* comp, FWProxyBuilderBase* pb );
   void addDashedArrow( double phi, double size, TEveElement* comp, FWProxyBuilderBase* pb );
   void addDashedLine( double phi, double theta, double size, TEveElement* comp, FWProxyBuilderBase* pb );
   void addDoubleLines( double phi, TEveElement* comp, FWProxyBuilderBase* pb );
   void drawEnergyScaledBox3D( const float* corners, float scale, TEveElement*,  FWProxyBuilderBase*, bool invert = false );
   void drawEtScaledBox3D( const float* corners, float energy, float maxEnergy, TEveElement*,  FWProxyBuilderBase*, bool invert = false );
   void drawEnergyTower3D( const float* corners, float scale, TEveElement*, FWProxyBuilderBase*, bool reflect = false );
   void drawEtTower3D( const float* corners, float scale, TEveElement*, FWProxyBuilderBase*, bool reflect = false );
}

#endif
