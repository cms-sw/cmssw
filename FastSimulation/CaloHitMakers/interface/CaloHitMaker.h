#ifndef CaloHitMaker_h
#define CaloHitMaker_h

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "Math/GenVector/Plane3D.h"
#include "SimG4CMS/Calo/interface/CaloHitID.h"

//CLHEP headers
//#include "CLHEP/Geometry/Point3D.h"
//#include "CLHEP/Geometry/Plane3D.h"


//STL headers
#include <string>
#include <map>

class CaloGeometryHelper;
class CalorimeterProperties;

class CaloHitMaker
{
 public:

  typedef math::XYZVector XYZVector;
  typedef math::XYZVector XYZPoint;
  typedef ROOT::Math::Plane3D Plane3D;

  CaloHitMaker(const CaloGeometryHelper * calo,DetId::Detector det,int subdetn,int cal,unsigned sht=0);
  virtual ~CaloHitMaker(){;}
  
  virtual bool addHit(double r,double phi,unsigned layer=0)=0;
  virtual void setSpotEnergy(double e)=0;
  virtual const std::map<CaloHitID,float>& getHits()=0; 

  const CaloGeometryHelper * getCalorimeter() const 
    {
      //      std::cout << "CaloHitMaker is returning myCalorimeter " << myCalorimeter << std::endl;
      return myCalorimeter;
    }

 protected:
  /// computes the intersection between a straight line defined by a & b
  /// and a plan
  static XYZPoint intersect(const Plane3D& p,const XYZPoint& a,const XYZPoint& b,double& t,bool segment,bool debug=false);

  const CaloGeometryHelper * myCalorimeter;    
  const CalorimeterProperties * theCaloProperties;
  double moliereRadius;
  double interactionLength;
  double spotEnergy;

  bool EMSHOWER;
  bool HADSHOWER;
  bool MIP;

 private:
  DetId::Detector base_;
  int subdetn_;
  int onCal_;


 protected:
  unsigned showerType_;
  std::map<CaloHitID,float> hitMap_;
  
};

#endif
