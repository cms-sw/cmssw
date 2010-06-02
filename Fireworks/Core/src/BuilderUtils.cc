#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include <math.h>
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "TGeoBBox.h"
#include "TGeoArb8.h"
#include "TColor.h"
#include "TROOT.h"
#include "TEveTrans.h"
#include "TEveGeoNode.h"
#include <time.h>
#include "DataFormats/FWLite/interface/Event.h"

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

void fw::addRhoZEnergyProjection( TEveElement* container,
                                  double r_ecal, double z_ecal,
                                  double theta_min, double theta_max,
                                  double phi,
                                  Color_t color)
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
   TColor* c = gROOT->GetColor( color );
   Float_t rgba[4] = { 1, 0, 0, 1 };
   if (c) {
      rgba[0] = c->GetRed();
      rgba[1] = c->GetGreen();
      rgba[2] = c->GetBlue();
   }

   if ( fabs(r2 - r1) > 1 ) {
      TGeoBBox *sc_box = new TGeoBBox(0., fabs(r2-r1)/2, 1);
      TEveGeoShape *element = new TEveGeoShape("r-segment");
      element->SetShape(sc_box);
      TEveTrans &t = element->RefMainTrans();
      t(1,4) = 0;
      t(2,4) = (r2+r1)/2;
      t(3,4) = fabs(z2)>fabs(z1) ? z2 : z1;

      element->SetPickable(kTRUE);
      element->SetMainColorRGB(rgba[0],rgba[1],rgba[2]);
      container->AddElement(element);
   }
   if ( fabs(z2 - z1) > 1 ) {
      TGeoBBox *sc_box = new TGeoBBox(0., 1, (z2-z1)/2);
      TEveGeoShape *element = new TEveGeoShape("z-segment");
      element->SetShape(sc_box);
      TEveTrans &t = element->RefMainTrans();
      t(1,4) = 0;
      t(2,4) = fabs(r2)>fabs(r1) ? r2 : r1;
      t(3,4) = (z2+z1)/2;

      element->SetMainColorRGB(rgba[0],rgba[1],rgba[2]);
      element->SetPickable(kTRUE);
      container->AddElement(element);
   }
}

TEveElementList *fw::getEcalCrystals (const EcalRecHitCollection *hits,
                                      const DetIdToMatrix &geo,
                                      double eta, double phi,
                                      int n_eta, int n_phi)
{
   std::vector<DetId> v;
   int ieta = (int)rint(eta / 1.74e-2);
   // black magic for phi
   int iphi = (int)rint(phi / 1.74e-2);
   if (iphi < 0)
      iphi = 360 + iphi;
   iphi += 10;
   for (int i = ieta - n_eta; i < ieta + n_eta; ++i) {
      for (int j = iphi - n_phi; j < iphi + n_phi; ++j) {
         if (EBDetId::validDetId(i, j % 360)) {
            v.push_back(EBDetId(i, j % 360));
//                  printf("pushing back (%d, %d)\n", i, j % 360);
         }
      }
   }
   return getEcalCrystals(hits, geo, v);
}

TEveElementList *fw::getEcalCrystals (const EcalRecHitCollection *hits,
                                      const DetIdToMatrix &geo,
                                      const std::vector<DetId> &detids)
{
   TEveGeoManagerHolder gmgr(TEveGeoShape::GetGeoMangeur());
   TEveElementList *ret = new TEveElementList("Ecal crystals");
   for (std::vector<DetId>::const_iterator k = detids.begin();
        k != detids.end(); ++k) {
      double size = 0 * 0.001; // default size
      if (hits != 0) {
         EcalRecHitCollection::const_iterator hit = hits->find(*k);
         if (hit != hits->end())
            size = hit->energy();
      }

      TEveGeoShape* egs = geo.getShape(k->rawId());
      // printf("1 egs  %d \n", egs->GetShape()->GetUniqueID());
      assert(egs != 0);
      TEveTrans &t = egs->RefMainTrans();
      t.MoveLF(3, -size / 2);
      TGeoShape* crystal_shape = 0;
      if ( const TGeoTrap* shape = dynamic_cast<const TGeoTrap*>(egs->GetShape())) {
         double scale = size*0.5/shape->GetDz();
         crystal_shape = new TGeoTrap( size/2,
                                       shape->GetTheta(), shape->GetPhi(),
                                       shape->GetH1()*scale + shape->GetH2()*(1-scale),
                                       shape->GetBl1()*scale + shape->GetBl2()*(1-scale),
                                       shape->GetTl1()*scale + shape->GetTl2()*(1-scale),
                                       shape->GetAlpha1(),
                                       shape->GetH2(), shape->GetBl2(), shape->GetTl2(),
                                       shape->GetAlpha2());
      }
      if ( !crystal_shape ) crystal_shape = new TGeoBBox(1.1, 1.1, size / 2, 0);

      egs->SetShape(crystal_shape);
      Float_t rgba[4] = { 1, 0, 0, 1 };
      egs->SetMainColorRGB(rgba[0], rgba[1], rgba[2]);
      egs->SetRnrSelf(true);
      egs->SetRnrChildren(true);
      ret->AddElement(egs);
   }
   return ret;
}

std::string
fw::getTimeGMT( const fwlite::Event& event )
{
   time_t t(event.time().value() >> 32);
   std::string text( asctime( gmtime(&t) ) );
   size_t pos = text.find('\n');
   if ( pos != std::string::npos ) text = text.substr(0,pos);
   text += " GMT";
   return text;
}

std::string
fw::getLocalTime( const fwlite::Event& event )
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
