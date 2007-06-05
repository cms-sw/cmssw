#ifndef Geometry_TrackerNumberingBuilder_GeometricDet_H
#define Geometry_TrackerNumberingBuilder_GeometricDet_H

#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Base/interface/DDRotationMatrix.h"
#include "DetectorDescription/Base/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <vector>
#include "FWCore/ParameterSet/interface/types.h"

//class DetId;
class DDFilteredView;

/**
 * Composite class GeometricDet. A composite can contain other composites, and so on;
 * You can understand what you are looking at via enum.
 */

class GeometricDet {
 public:
  
  typedef std::vector< GeometricDet const *>  ConstGeometricDetContainer;
  typedef std::vector< GeometricDet const *>  GeometricDetContainer;
  typedef DDExpandedView::nav_type nav_type;
  typedef Surface::PositionType Position;
  typedef Surface::RotationType Rotation;

  //
  // more can be added; please add at the end!
  //
  typedef enum GDEnumType {unknown=100, Tracker=0, PixelBarrel=1, PixelEndCap=2,
			  TIB=3, TID=4, TOB=5, TEC=6,
			  layer=8, wheel=9, strng=10, rod=11,petal=12,ring=13,
			  ladder=14, mergedDet=15, DetUnit=16,disk=17, panel=18 } GeometricEnumType;
  /**
   * Constructors to be used when looping over DDD
   */
  GeometricDet(nav_type navtype, GeometricEnumType dd);

  GeometricDet(DDExpandedView* ev, GeometricEnumType dd);
  GeometricDet(DDFilteredView* fv, GeometricEnumType dd);
  
  /*
  GeometricDet(const GeometricDet &);

  GeometricDet & operator=( const GeometricDet & );
  */

  /**
   * set or add or clear components
   */
  void setGeographicalID(DetId id){_geographicalID = id;}
  void setComponents(GeometricDetContainer const & cont) {_container = cont;}
  void addComponents(GeometricDetContainer const & cont);
  void addComponent(GeometricDet*);
  /**
   * clearComponents() only empties the container, the components are not deleted!
   */
  void clearComponents(){_container.resize(0);}
  /**
   * deleteComponents() explicitly deletes the daughters
   */
  void deleteComponents();
  /**
   * deepDeleteComponents() traverses the treee and deepDeletes() all of them.
   * Is  this a final cleanup?
   */
  void deepDeleteComponents();
  
  bool isLeaf() const {return (_container.size() == 0);}
  
  /**
   * Access methods
   */
  DDRotationMatrix const & rotation() const {return _rot;}
  DDTranslation const & translation() const {return _trans;}
  DDSolidShape const & shape() const  {return _shape;}
  GeometricEnumType type() const {return _type;}
  DDName const & name() const {return _ddname;};
  nav_type navType() const {return _ddd;}
  std::vector<double> const & params() const {return _params;}


  virtual ~GeometricDet();
  
  /**
   * components() returns explicit components; please note that in case of a leaf 
   * GeometricDet it returns nothing (an empty vector)
   */
  virtual GeometricDetContainer const & components() const {
    return _container;
  }

  /**
   * deepComponents() returns all the components below; please note that 
   * if the current GeometricDet is a leaf, it returns it!
   */

  ConstGeometricDetContainer deepComponents() const;
  void deepComponents(ConstGeometricDetContainer &) const;

  //rr
  /** parents() retuns the geometrical history
   */
  virtual std::vector< DDExpandedNode > const &  parents() const {return _parents;}
  //rr  
  
  /**
   *geometricalID() returns the ID associated to the GeometricDet.
   */
  virtual DetId geographicalID() const ;

  /**
   *positionBounds() returns the position in cm. 
   */
  virtual Position positionBounds() const; 

  /**
   *rotationBounds() returns the rotation matrix. 
   */
  virtual Rotation  rotationBounds() const; 

  /**
   *bounds() returns the Bounds.
   */
  virtual const Bounds * bounds() const; 
  
  virtual int copyno() const {return _copy;}
  virtual double volume() const {return _volume;}
  virtual double density() const {return _density;}
  virtual double weight() const {return _weight;}
  virtual std::string const &  material() const {return _material;}
  
 private:

  GeometricDetContainer _container;
  DDTranslation _trans;
  DDRotationMatrix _rot;
  DDSolidShape _shape;
  nav_type _ddd;
  DDName _ddname;
  GeometricEnumType _type;
  std::vector<double> _params;
  DetId _geographicalID;

  std::vector< DDExpandedNode > _parents;
  double _volume;
  double _density;
  double _weight;
  int    _copy;
  std::string _material;
};

#endif

