#include "FastSimulation/CaloHitMakers/interface/CaloHitMaker.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloGeometryHelper.h"
#include "FastSimulation/CalorimeterProperties/interface/CalorimeterProperties.h"
#include "FastSimulation/CalorimeterProperties/interface/PreshowerProperties.h"
#include "FastSimulation/CalorimeterProperties/interface/PreshowerLayer1Properties.h"
#include "FastSimulation/CalorimeterProperties/interface/ECALProperties.h"
#include "FastSimulation/CalorimeterProperties/interface/HCALProperties.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

typedef ROOT::Math::Plane3D::Point Point;

CaloHitMaker::CaloHitMaker(const CaloGeometryHelper * theCalo,DetId::Detector basedet,int subdetn,int cal,unsigned sht)
  :myCalorimeter(theCalo),theCaloProperties(nullptr),base_(basedet),subdetn_(subdetn),onCal_(cal),showerType_(sht)
{
  //  std::cout << " FamosCalorimeter " << basedet << " " << cal << std::endl;
  EMSHOWER=(sht==0);
  HADSHOWER=(sht==1);
  MIP=(sht==2);
  if(base_==DetId::Ecal&&(subdetn_==EcalBarrel||subdetn==EcalEndcap)&&onCal_)
    theCaloProperties = myCalorimeter->ecalProperties(onCal_);
  // is it really necessary to cast here ? 
  if(base_==DetId::Ecal&&subdetn_==EcalPreshower&&onCal_)
    theCaloProperties = myCalorimeter->layer1Properties(onCal_);
  if(base_==DetId::Hcal&&cal) theCaloProperties = myCalorimeter->hcalProperties(onCal_);

  if(theCaloProperties)
    {
      moliereRadius=theCaloProperties->moliereRadius();
      interactionLength=theCaloProperties->interactionLength();
    }
  else
    {
      moliereRadius=999;
      interactionLength=999;
    }
}


CaloHitMaker::XYZPoint 
CaloHitMaker::intersect(const Plane3D& p,const XYZPoint& a,const XYZPoint& b, 
			double& t,bool segment, bool debug) 
{
  t=-9999.;
  // En Attendant //
  XYZVector normal = p.Normal();
  double AAA = normal.X();
  double BBB = normal.Y();
  double CCC = normal.Z();
  //  double DDD = p.Distance(Point(0.,0.,0.));
  double DDD = p.HesseDistance();
  //  double denom = p.A()*(b.X()-a.X()) + p.B()*(b.Y()-a.Y()) + p.C()*(b.Z()-a.Z());
  double denom = AAA*(b.X()-a.X()) + BBB*(b.Y()-a.Y()) + CCC*(b.Z()-a.Z());
  if(denom!=0.)
    {
      // t=-(p.A()*a.X()+p.B()*a.Y()+p.C()*a.Z()+p.D());
      t=-(AAA*a.X()+BBB*a.Y()+CCC*a.Z()+DDD);
      t/=denom;
      if(debug) std::cout << " T = " << t <<std::endl; 
      if(segment)
	{
	  if(t>=0&&t<=1)
	    return XYZPoint(a.X()+(b.X()-a.X())*t,
			    a.Y()+(b.Y()-a.Y())*t,
			    a.Z()+(b.Z()-a.Z())*t);      
	}
      else
	{
	  return XYZPoint(a.X()+(b.X()-a.X())*t,
			  a.Y()+(b.Y()-a.Y())*t,
			  a.Z()+(b.Z()-a.Z())*t);      
	}
	  
     
    }

  return XYZPoint(0.,0.,0.);
}
