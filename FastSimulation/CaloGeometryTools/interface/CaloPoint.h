#ifndef FastSimulation_CaloGeometryTools_CaloPoint_h
#define FastSimulation_CaloGeometryTools_CaloPoint_h
/*
 *
 * A point belonging to a given detector
 * 
 */


#include "Geometry/CaloTopology/interface/CaloDirection.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Math/interface/Vector3D.h"

#include <string>

//ideally this class should inherit from XYZPoint & CellID 

class CaloPoint : public math::XYZVector
{

 public: 

  typedef math::XYZVector XYZVector;
  typedef math::XYZVector XYZPoint;

  /// Empty constructor
  CaloPoint():XYZPoint(){;};
//  /// Constructor from DetId, side and position. 
//  CaloPoint(DetId cell, CaloDirection side, const XYZPoint& position);
//
//  /// Constructor side and position
//  CaloPoint( CaloDirection side, const XYZPoint& position):XYZPoint(position),side_(side){;};

  /// constructor for ECAL
  CaloPoint(const  DetId& cell, CaloDirection side, const XYZPoint& position);

  /// constructor for HCAL
  CaloPoint(DetId::Detector detector,const XYZPoint& position);

  /// constructor for preshower
  CaloPoint(DetId::Detector detector,int subdetn,int layer, const XYZPoint & position);

  ~CaloPoint(){;}
  /// returns the cellID
  inline DetId getDetId() const {return cellid_;};
  /// returns the Side (see numbering)
  inline CaloDirection getSide() const {return side_;};

  inline bool operator<(const CaloPoint & p) const
    {return this->mag2()<p.mag2() ;};
      
  inline void setDetId(DetId::Detector det) {detector_=det;}
  inline DetId::Detector whichDetector() const {return detector_;};

  inline void setSubDetector(int i) {subdetector_=i;}

  ///  watch out, only valid in ECAL and preshower 
  inline int whichSubDetector() const {return subdetector_;};

  inline void setLayer(int i) {layer_=i;}
  
  inline int whichLayer() const {return layer_;}

	    //  const CaloGeometryHelper * getCalorimeter() const { return myCalorimeter_;}

  private:
	    //  const CaloGeometryHelper * myCalorimeter_;
  DetId cellid_;
  CaloDirection side_;
  DetId::Detector detector_;
  int subdetector_;
  int layer_;


 public:
  class DistanceToVertex
    {
    public:
      DistanceToVertex(const XYZPoint & vert):vertex(vert) {};
      ~DistanceToVertex(){};
      bool operator() (const CaloPoint& point1,const CaloPoint& point2)
	{
	  return ((point1-vertex).mag2()<(point2-vertex).mag2());
	}
    private:
      XYZPoint vertex;
    };
};
#include <iosfwd>
std::ostream& operator <<(std::ostream& o , const CaloPoint& cid);

#endif
