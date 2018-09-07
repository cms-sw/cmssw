#ifndef Geometry_MTDNumberingBuilder_GeometricTimingDet_H
#define Geometry_MTDNumberingBuilder_GeometricTimingDet_H

#include "CondFormats/GeometryObjects/interface/PGeometricTimingDet.h"
#include "DetectorDescription/Core/interface/DDExpandedView.h"
#include "DetectorDescription/Core/interface/DDRotationMatrix.h"
#include "DetectorDescription/Core/interface/DDTranslation.h"
#include "DetectorDescription/Core/interface/DDSolidShapes.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/GeometrySurface/interface/Bounds.h"
#include "DataFormats/DetId/interface/DetId.h"

#include <vector>
#include <memory>
#include "FWCore/ParameterSet/interface/types.h"

#include <ext/pool_allocator.h>

class DDFilteredView;

/**
 * Composite class GeometricTimingDet. A composite can contain other composites, and so on;
 * You can understand what you are looking at via enum.
 */

class GeometricTimingDet {
 public:

  typedef DDExpandedView::nav_type DDnav_type;
  typedef DDExpandedView::NavRange NavRange;

  typedef std::vector< GeometricTimingDet const *>  ConstGeometricTimingDetContainer;
  typedef std::vector< GeometricTimingDet *>  GeometricTimingDetContainer;

#ifdef PoolAlloc  
  typedef std::vector< DDExpandedNode, PoolAlloc<DDExpandedNode> > GeoHistory;
  typedef std::vector<int, PoolAlloc<int> > nav_type;
#else
  typedef std::vector<DDExpandedNode> GeoHistory;
  typedef DDExpandedView::nav_type nav_type;
#endif
  typedef Surface::PositionType Position;
  typedef Surface::RotationType Rotation;

  //
  // more can be added; please add at the end!
  //
  typedef enum GTDEnumType {unknown=100, MTD=0, BTL=1, BTLLayer=2, BTLTray=3, 
                            BTLModule=4, BTLSensor=5, BTLCrystal=6,
                            ETL=7, ETLDisc=8, ETLRing=9, ETLModule=10, 
                            ETLSensor=11 } GeometricTimingEnumType;

  /**
   * Constructors to be used when looping over DDD
   */
#ifdef GEOMETRICDETDEBUG
  GeometricTimingDet(DDnav_type const & navtype, GeometricTimingEnumType dd);
  GeometricTimingDet(DDExpandedView* ev, GeometricTimingEnumType dd);
#endif
  GeometricTimingDet(DDFilteredView* fv, GeometricTimingEnumType dd);
  GeometricTimingDet(const PGeometricTimingDet::Item& onePGD, GeometricTimingEnumType dd);
    

  /**
   * set or add or clear components
   */
  void setGeographicalID(DetId id) {
    geographicalID_ = id; 
    //std::cout <<"setGeographicalID " << int(id) << std::endl;
  }
#ifdef GEOMETRICDETDEBUG
  void setComponents(GeometricTimingDetContainer const & cont) {
    container_ = cont; 
    //std::cout <<"setComponents" << std::endl;
  }
#endif
  void addComponents(GeometricTimingDetContainer const & cont);
  void addComponents(ConstGeometricTimingDetContainer const & cont);
  void addComponent(GeometricTimingDet*);
  /**
   * clearComponents() only empties the container, the components are not deleted!
   */
  void clearComponents() {
    container_.clear();
  }
 
  /**
   * deleteComponents() explicitly deletes the daughters
   * 
   */
  void deleteComponents();

  bool isLeaf() const { 
    return container_.empty(); 
  }
  
  GeometricTimingDet* component(size_t index) {
    return const_cast<GeometricTimingDet*>(container_[index]);
  }

  /**
   * Access methods
   */
  DDRotationMatrix const & rotation() const {
    return rot_;
  }
  DDTranslation const & translation() const {
    return trans_;
  }
  double phi() const { 
    return phi_; 
  }
  double rho() const { 
    return rho_; 
  }

  DDSolidShape const & shape() const  {
    return shape_;
  }
  GeometricTimingEnumType type() const {
    return type_;
  }
  DDName const & name() const {
    return ddname_;
  }
  // internal representaion
  nav_type const & navType() const {
    return ddd_;
  }
  // representation neutral interface
  NavRange navRange() const {
    return NavRange(&ddd_.front(),ddd_.size());
  }
  // more meaningfull name (maybe)
  NavRange navpos() const {
    return NavRange(&ddd_.front(),ddd_.size());
  }
  std::vector<double> const & params() const {
    //std::cout<<"params"<<std::endl;
    return params_;
  }

  ~GeometricTimingDet();
  
  /**
   * components() returns explicit components; please note that in case of a leaf 
   * GeometricTimingDet it returns nothing (an empty vector)
   */
  ConstGeometricTimingDetContainer & components() {
    return container_;
  }  
  ConstGeometricTimingDetContainer const & components() const {
    return container_;
  }

  /**
   * deepComponents() returns all the components below; please note that 
   * if the current GeometricTimingDet is a leaf, it returns it!
   */

  ConstGeometricTimingDetContainer deepComponents() const;
  void deepComponents(ConstGeometricTimingDetContainer & cont) const;

#ifdef GEOMETRICDETDEBUG
  /** parents() retuns the geometrical history
   * mec: only works if this is built from DD and not from reco DB.
   */
  GeoHistory const &  parents() const {
    return parents_;
  }
  //rr  
#endif
  
  /**
   *geometricalID() returns the ID associated to the GeometricTimingDet.
   */
  DetId geographicalID() const  { 
    return geographicalID_; 
  }
  DetId geographicalId() const  { 
    return geographicalID_; 
  }

  /**
   *positionBounds() returns the position in cm. 
   */
  Position positionBounds() const; 

  /**
   *rotationBounds() returns the rotation matrix. 
   */
  Rotation  rotationBounds() const; 

  /**
   *bounds() returns the Bounds.
   */
  std::unique_ptr<Bounds> bounds() const; 
#ifdef GEOMETRICDETDEBUG
  int copyno() const {
    return copy_;
  }
  double volume() const {
    return volume_;
  }
  double density() const {
    return density_;
  }
  double weight() const {
    return weight_;
  }
  std::string const &  material() const {
    return material_;
  }
#endif
  double radLength() const {
    return radLength_;
  }
  double xi() const {
    return xi_;
  }
  /**
   * The following four pix* methods only return meaningful results for pixels.
   */
  double pixROCRows() const {
    return pixROCRows_;
  }
  double pixROCCols() const {
    return pixROCCols_;
  }
  double pixROCx() const {
    return pixROCx_;
  }
  double pixROCy() const {
    return pixROCy_;
  }

  /**
   * The following two are only meaningful for the silicon tracker.
   */  
  bool stereo() const {
    return stereo_;
  }
  double siliconAPVNum() const {
    return siliconAPVNum_;
  }

  /**
   * what it says... used the DD in memory model to build the geometry... or not.
   */
#ifdef GEOMETRICDETDEBUG
  bool wasBuiltFromDD() const {
    return fromDD_;
  }
#endif  

 private:

  ConstGeometricTimingDetContainer container_;
  DDTranslation trans_;
  double phi_;
  double rho_;
  DDRotationMatrix rot_;
  DDSolidShape shape_;
  nav_type ddd_;
  DDName ddname_;
  GeometricTimingEnumType type_;
  std::vector<double> params_;

  DetId geographicalID_;
#ifdef GEOMETRICDETDEBUG
  GeoHistory parents_;
  double volume_;
  double density_;
  double weight_;
  int    copy_;
  std::string _material;
#endif
  double radLength_;
  double xi_;
  double pixROCRows_;
  double pixROCCols_;
  double pixROCx_;
  double pixROCy_;
  bool stereo_;
  double siliconAPVNum_;
#ifdef GEOMETRICDETDEBUG
  bool fromDD_;
#endif

};

#undef PoolAlloc
#endif

