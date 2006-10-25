#include "FastSimulation/CaloHitMakers/interface/CaloHitMaker.h"
#include "FastSimulation/CaloGeometryTools/interface/CaloGeometryHelper.h"
#include "FastSimulation/CalorimeterProperties/interface/CalorimeterProperties.h"
#include "FastSimulation/CalorimeterProperties/interface/PreshowerProperties.h"
#include "FastSimulation/CalorimeterProperties/interface/HCALProperties.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"


CaloHitMaker::CaloHitMaker(const CaloGeometryHelper * theCalo,DetId::Detector basedet,int subdetn,int cal,unsigned sht)
  :myCalorimeter(theCalo),theCaloProperties(NULL),base_(basedet),subdetn_(subdetn),onCal_(cal),showerType_(sht)
{
  //  std::cout << " FamosCalorimeter " << basedet << " " << cal << std::endl;
  EMSHOWER=(sht==0);
  HADSHOWER=(sht==1);
  if(base_==DetId::Ecal&&(subdetn_==EcalBarrel||subdetn==EcalEndcap)&&onCal_)
    theCaloProperties = (CalorimeterProperties*)myCalorimeter->ecalProperties(onCal_);
  // is it really necessary to cast here ? 
  if(base_==DetId::Ecal&&subdetn_==EcalPreshower&&onCal_)
    theCaloProperties = (PreshowerProperties*)myCalorimeter->layer1Properties(onCal_);
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


HepPoint3D CaloHitMaker::intersect(const HepPlane3D& p,const HepPoint3D& a,const HepPoint3D& b, double& t,bool segment, bool debug) 
{
  t=-9999.;
  double denom=p.a()*(b.x()-a.x()) + p.b()*(b.y()-a.y()) +p.c()*(b.z()-a.z());
  if(denom!=0.)
    {
      t=-(p.a()*a.x()+p.b()*a.y()+p.c()*a.z()+p.d());
      t/=denom;
      if(debug) std::cout << " T = " << t <<std::endl; 
      if(segment)
	{
	  if(t>=0&&t<=1)
	    return HepPoint3D(a.x()+(b.x()-a.x())*t,
			      a.y()+(b.y()-a.y())*t,
			      a.z()+(b.z()-a.z())*t);      
	}
      else
	{
	  return HepPoint3D(a.x()+(b.x()-a.x())*t,
			    a.y()+(b.y()-a.y())*t,
			    a.z()+(b.z()-a.z())*t);      
	}
	  
     
    }

  return HepPoint3D(0.,0.,0.);
}
