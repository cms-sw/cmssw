//-------------------------------------------------
//
/**  \class DTGeomSupplier
 *   Defines the ability to calculate coordinates 
 *   of L1DT Trigger objects
 *
 *   $Date: 2008/11/05 00:08:28 $
 *   $Revision: 1.5 $
 *
 *   \author C.Grandi
 */
//
//--------------------------------------------------
#ifndef DT_GEOM_SUPPLIER_H
#define DT_GEOM_SUPPLIER_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//----------------------
// Base Class Headers --
//----------------------
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "L1Trigger/DTUtilities/interface/DTTrigGeom.h"
#include "L1Trigger/DTUtilities/interface/DTTrigData.h"


//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTGeomSupplier {

 public:

  ///  Constructor
  DTGeomSupplier(DTTrigGeom* geom) : _geom(geom) {}

  /// Destructor
  virtual ~DTGeomSupplier() {}

  /// Associated geometry
  inline DTTrigGeom* geom() const { return _geom; }

  /// Associated chamber
  inline const DTChamber* stat() const { return _geom->stat(); }

  /// Identifier of the associated chamber
  inline DTChamberId ChamberId() const { return _geom->statId(); }

   /// Return wheel number
  inline int wheel() const { return _geom->wheel(); }

  /// Return station number
  inline int station() const { return _geom->station(); }

  /// Return sector number
  inline int sector() const { return _geom->sector(); }

  /// Local position in chamber of a trigger-data object
  virtual LocalPoint localPosition(const DTTrigData*) const = 0;

  /// Local direction in chamber of a trigger-data object
  virtual LocalVector localDirection(const DTTrigData*) const = 0;

  /// CMS position in chamber of a trigger-data object
  inline GlobalPoint CMSPosition(const DTTrigData* trig) const {
    return _geom->toGlobal(localPosition(trig));
  }

  /// CMS direction in chamber of a trigger -data object
  inline GlobalVector CMSDirection(const DTTrigData* trig) const {
    return _geom->toGlobal(localDirection(trig));
  }

  /// Print a trigger-data object with also local and global position/direction
  virtual void print(const DTTrigData* trig) const;

protected:

  DTTrigGeom* _geom;

};

#endif
