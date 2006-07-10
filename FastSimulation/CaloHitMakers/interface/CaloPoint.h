#ifndef CaloPoint_h
#define CaloPoint_h
/*
 *
 * A point belonging to a given detector
 * 
 */

#include "FastSimulation/CalorimeterProperties/interface/Calorimeter.h"
#include "DataFormats/DetId/interface/DetId.h"
//CLHEP headers
#include "CLHEP/Geometry/Point3D.h"

#include <iostream>
#include <string>
//ideally this class should inherit from HepPoint3D & CellID 

class CaloPoint : public HepPoint3D
{

  public: 

  /// Empty constructor
  CaloPoint():HepPoint3D(){;};
  /// Constructor from CellID, side and position. No CaloDirection yet
    //  CaloPoint(DetId cell, CaloDirection side, const HepPoint3D& position);

  /// Constructor side and position
    //CaloPoint( CaloDirection side, const HepPoint3D& position):HepPoint3D(position),side_(side){;};

  /// an other detector for PS and HCAL
    //  CaloPoint(string detector,CaloDirection side, const HepPoint3D& position);

 CaloPoint(const Calorimeter * calo, std::string, const HepPoint3D& position);

  ~CaloPoint(){;}
  /// returns the cellID
  inline DetId getDetId() const {return cellid_;};
  /// returns the Side (see numbering)
    //  inline CaloDirection getSide() const {return side_;};

  inline bool operator<(const CaloPoint & p) const
    {return this->mag()<p.mag() ;};
      
  inline void setDetector(DetId::Detector det) {detector_=det;}
  inline std::string whichDetector() const {return detector_;};

  inline void setSubDetector(int i) {subdetector_=i;}
  inline int whichSubDetector() const {return subdetector_;};

  const Calorimeter * getCalorimeter() const { return myCalorimeter_;}

  private:
  const Calorimeter * myCalorimeter_;
  DetId cellid_;
  //  CaloDirection side_;
  std::string detector_;
  int subdetector_;



 public:
  class DistanceToVertex
    {
    public:
      DistanceToVertex(const   HepPoint3D & vert):vertex(vert) {};
      ~DistanceToVertex(){};
      bool operator() (const CaloPoint& point1,const CaloPoint& point2)
	{
	  return ((point1-vertex).mag()<(point2-vertex).mag());
	}
    private:
      HepPoint3D vertex;
    };
};
#include <iosfwd>
std::ostream& operator <<(std::ostream& o , const CaloPoint& cid);

#endif
