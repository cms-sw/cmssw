#include "Fireworks/Core/interface/BuilderUtils.h"
#include "Fireworks/Core/interface/DetIdToMatrix.h"
#include <math.h>
#include "TEveTrack.h"
#include "TEveTrackPropagator.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"
#include "Fireworks/Core/interface/FWEventItem.h"
#include "TEveGeoShapeExtract.h"
#include "TGeoBBox.h"
#include "TGeoArb8.h"
#include "TColor.h"
#include "TROOT.h"
#include "TEveTrans.h"
#include "TEveGeoNode.h"
#include "TEveStraightLineSet.h"

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

TEveTrack* fw::getEveTrack( const reco::Track& track,
			    double max_r /* = 120 */,
			    double max_z /* = 300 */,
			    double magnetic_field /* = 4 */ )
{
   TEveTrackPropagator *propagator = new TEveTrackPropagator();
   propagator->SetMagField( - magnetic_field );
   propagator->SetMaxR( max_r );
   propagator->SetMaxZ( max_z );

   TEveRecTrack t;
   t.fBeta = 1.;
   t.fP = TEveVector( track.px(), track.py(), track.pz() );
   t.fV = TEveVector( track.vx(), track.vy(), track.vz() );
   t.fSign = track.charge();
   TEveTrack* trk = new TEveTrack(&t, propagator);
   trk->MakeTrack();
   return trk;
}

std::string fw::NamedCounter::str() const
{
   std::stringstream s;
   s << m_name << m_index;
   return s.str();
}

TEveGeoShapeExtract* fw::getShapeExtract( const char* name,
					  TGeoBBox* shape,
					  Color_t color )
{
   TEveGeoShapeExtract* extract = new TEveGeoShapeExtract(name);
   TColor* c = gROOT->GetColor(color);
   Float_t rgba[4] = { 1, 0, 0, 1 };
   if (c) {
      rgba[0] = c->GetRed();
      rgba[1] = c->GetGreen();
      rgba[2] = c->GetBlue();
   }
   extract->SetRGBA(rgba);
   extract->SetRnrSelf(true);
   extract->SetRnrElements(true);
   extract->SetShape(shape);
   return extract;
}

void fw::addRhoZEnergyProjection( TEveElement* container,
			      double r_ecal, double z_ecal, 
			      double theta_min, double theta_max, 
			      double phi,
			      Color_t color)
{
   
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
      TEveTrans t;
      t(1,4) = 0; 
      t(2,4) = (r2+r1)/2;
      t(3,4) = fabs(z2)>fabs(z1) ? z2 : z1;
      TEveGeoShapeExtract *extract = new TEveGeoShapeExtract("r-segment");
      extract->SetTrans(t.Array());
      extract->SetRGBA(rgba);
      extract->SetRnrSelf(true);
      extract->SetRnrElements(true);
      extract->SetShape(sc_box);
      TEveElement* element = TEveGeoShape::ImportShapeExtract(extract, 0);
      element->SetPickable(kTRUE);
      container->AddElement(element);
   }
   if ( fabs(z2 - z1) > 1 ) {
      TGeoBBox *sc_box = new TGeoBBox(0., 1, (z2-z1)/2);
      TEveTrans t;
      t(1,4) = 0; 
      t(2,4) = fabs(r2)>fabs(r1) ? r2 : r1;
      t(3,4) = (z2+z1)/2;
      TEveGeoShapeExtract *extract = new TEveGeoShapeExtract("z-segment");
      extract->SetTrans(t.Array());
      extract->SetRGBA(rgba);
      extract->SetRnrSelf(true);
      extract->SetRnrElements(true);
      extract->SetShape(sc_box);
      TEveElement* element = TEveGeoShape::ImportShapeExtract(extract, 0);
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
// 		    printf("pushing back (%d, %d)\n", i, j % 360);
	       }
	  }
     }
     return getEcalCrystals(hits, geo, v);
}

TEveElementList *fw::getEcalCrystals (const EcalRecHitCollection *hits,
				      const DetIdToMatrix &geo,
				      const std::vector<DetId> &detids)
{
  TEveElementList *ret = new TEveElementList("Ecal crystals");
  for (std::vector<DetId>::const_iterator k = detids.begin();
       k != detids.end(); ++k) {
    double size = 0 * 0.001;  // default size
    if (hits != 0) {
      EcalRecHitCollection::const_iterator hit = hits->find(*k);
      if (hit != hits->end())
	size = hit->energy();
    }
    TEveGeoShapeExtract* extract = geo.getExtract(k->rawId());
    assert(extract != 0);
    TEveTrans t = extract->GetTrans();
    t.MoveLF(3, - size / 2);
    // TGeoBBox *sc_box = new TGeoBBox(1.1, 1.1, size / 2, 0);
    TGeoShape* crystal_shape = 0;
    if ( const TGeoTrap* shape = dynamic_cast<const TGeoTrap*>(extract->GetShape())) {
      double scale = size/2/shape->GetDz();
      crystal_shape = new TGeoTrap( size/2,
				    shape->GetTheta(), shape->GetPhi(),
				    shape->GetH1()*scale + shape->GetH2()*(1-scale),
				    shape->GetBl1()*scale + shape->GetBl2()*(1-scale),
				    shape->GetTl1()*scale + shape->GetTl2()*(1-scale),
				    shape->GetAlpha1(),
				    shape->GetH2(), shape->GetBl2(), shape->GetTl2(),
				    shape->GetAlpha2());
    }
    if ( ! crystal_shape ) crystal_shape = new TGeoBBox(1.1, 1.1, size / 2, 0);
    TEveGeoShapeExtract *extract2 = new TEveGeoShapeExtract("SC");
    extract2->SetTrans(t.Array());
    Float_t rgba[4] = { 1, 0, 0, 1 };
    extract2->SetRGBA(rgba);
    extract2->SetRnrSelf(true);
    extract2->SetRnrElements(true);
    extract2->SetShape(crystal_shape);
    ret->AddElement(TEveGeoShape::ImportShapeExtract(extract2,0));
  }
  return ret;
}

void fw::addStraightLineSegment( TEveStraightLineSet * marker,
				 reco::Candidate const * cand,
				 double scale_factor)
{  
  double phi = cand->phi();
  double theta = cand->theta();
  double size = cand->pt() * scale_factor;
  marker->AddLine( 0, 0, 0, size * cos(phi)*sin(theta), size *sin(phi)*sin(theta), size*cos(theta));
}

/*TEveElementList *fw::getMuonCalTowers (double eta, double phi) 
  {
  // Input muon eta, phi and return towers within certain radius of muon object
  // Well, shit.  That ain't going to work...
  
  TEveElementList *ret = new TEveElementList("ECAL Towers");
  
  const CaloTowerCollection* towers=0;
  m_item->get(towers);
  if(0==towers) {
  std::cout <<"Failed to get CaloTowers"<<std::endl;
  return;
  }
  
  }
*/
