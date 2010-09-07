#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/FWProxyBuilderBase.h"
#include <math.h>

#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "FWCore/Common/interface/EventBase.h"

#include "TGeoBBox.h"
#include "TGeoArb8.h"
#include "TColor.h"
#include "TROOT.h"
#include "TEveTrans.h"
#include "TEveGeoNode.h"
#include <time.h>

std::pair<double,double> fw::getPhiRange( const std::vector<double>& phis, double phi )
{
   double min =  100;
   double max = -100;

   for ( std::vector<double>::const_iterator i = phis.begin();
         i != phis.end(); ++i )
   {
      double aphi = *i;
      // make phi continuous around jet phi
      if ( aphi - phi > M_PI ) aphi -= 2*M_PI;
      if ( phi - aphi > M_PI ) aphi += 2*M_PI;
      if ( aphi > max ) max = aphi;
      if ( aphi < min ) min = aphi;
   }

   if ( min > max ) return std::pair<double,double>(0,0);

   return std::pair<double,double>(min,max);
}

std::string fw::NamedCounter::str() const
{
   std::stringstream s;
   s << m_name << m_index;
   return s.str();
}

TEveGeoShape* fw::getShape( const char* name,
                            TGeoBBox* shape,
                            Color_t color )
{
   TEveGeoShape* egs = new TEveGeoShape(name);
   TColor* c = gROOT->GetColor(color);
   Float_t rgba[4] = { 1, 0, 0, 1 };
   if (c) {
      rgba[0] = c->GetRed();
      rgba[1] = c->GetGreen();
      rgba[2] = c->GetBlue();
   }
   egs->SetMainColorRGB(rgba[0], rgba[1], rgba[2]);
   egs->SetShape(shape);
   return egs;
}

void fw::addRhoZEnergyProjection( FWProxyBuilderBase* pb, TEveElement* container,
                                  double r_ecal, double z_ecal,
                                  double theta_min, double theta_max,
                                  double phi)
{
   TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
   double z1 = r_ecal/tan(theta_min);
   if ( z1 > z_ecal ) z1 = z_ecal;
   if ( z1 < -z_ecal ) z1 = -z_ecal;
   double z2 = r_ecal/tan(theta_max);
   if ( z2 > z_ecal ) z2 = z_ecal;
   if ( z2 < -z_ecal ) z2 = -z_ecal;
   double r1 = z_ecal*fabs(tan(theta_min));
   if ( r1 > r_ecal ) r1 = r_ecal;
   if ( phi < 0 ) r1 = -r1;
   double r2 = z_ecal*fabs(tan(theta_max));
   if ( r2 > r_ecal ) r2 = r_ecal;
   if ( phi < 0 ) r2 = -r2;


   if ( fabs(r2 - r1) > 1 ) {
      TGeoBBox *sc_box = new TGeoBBox(0., fabs(r2-r1)/2, 1);
      TEveGeoShape *element = new TEveGeoShape("r-segment");
      element->SetShape(sc_box);
      TEveTrans &t = element->RefMainTrans();
      t(1,4) = 0;
      t(2,4) = (r2+r1)/2;
      t(3,4) = fabs(z2)>fabs(z1) ? z2 : z1;
      pb->setupAddElement(element, container);
   }
   if ( fabs(z2 - z1) > 1 ) {
      TGeoBBox *sc_box = new TGeoBBox(0., 1, (z2-z1)/2);
      TEveGeoShape *element = new TEveGeoShape("z-segment");
      element->SetShape(sc_box);
      TEveTrans &t = element->RefMainTrans();
      t(1,4) = 0;
      t(2,4) = fabs(r2)>fabs(r1) ? r2 : r1;
      t(3,4) = (z2+z1)/2;
      pb->setupAddElement(element, container);
   }
}

std::string
fw::getTimeGMT(const edm::EventBase& event)
{
   time_t t(event.time().value() >> 32);
   std::string text( asctime( gmtime(&t) ) );
   size_t pos = text.find('\n');
   if ( pos != std::string::npos ) text = text.substr(0,pos);
   text += " GMT";
   return text;
}

std::string
fw::getLocalTime(const edm::EventBase& event)
{
   time_t t(event.time().value() >> 32);
   std::string text( asctime( localtime(&t) ) );
   size_t pos = text.find('\n');
   if ( pos != std::string::npos ) text = text.substr(0,pos);
   text += " ";
   if ( daylight )
     text += tzname[1];
   else
     text += tzname[0];
   return text;
}
