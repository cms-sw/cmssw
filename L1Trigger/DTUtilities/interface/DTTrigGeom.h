//-------------------------------------------------
//
/**  \class DTTrigGeom
 *     Muon Barrel Trigger Geometry
 *
 *
 *   $Date: 2006/07/19 10:32:51 $
 *   $Revision: 1.1 $
 *
 *   \author C.Grandi
 *   \modifications S.Vanini
 */
//
//--------------------------------------------------
#ifndef DT_TRIG_GEOM_H
#define DT_TRIG_GEOM_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//----------------------
// Base Class Headers --
//----------------------
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "Geometry/DTGeometry/interface/DTChamber.h"
#include "L1Trigger/DTUtilities/interface/DTConfig.h"
#include "L1Trigger/DTUtilities/interface/DTBtiId.h"
#include "L1Trigger/DTUtilities/interface/DTTracoId.h"
#include "Geometry/Vector/interface/LocalPoint.h" 
#include "Geometry/Vector/interface/GlobalPoint.h" 
//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTTrigGeom {

  public:

    /// Constructor
    DTTrigGeom(DTChamber* stat, DTConfig* conf);
  
    /// Destructor 
    ~DTTrigGeom();
 
    /// Associated chamber
    inline DTChamber* stat() const { return _stat; }

    /// Identifier of the associated chamber
    inline DTChamberId statId() const { return _stat->id(); }

    /// Return wheel number
    inline int wheel() const { return _stat->id().wheel(); }

    /// Return station number
    inline int station() const { return _stat->id().station(); }

    /// Return sector number
    inline int sector() const { return _stat->id().sector(); }

    /// Return the configuration set
    inline DTConfig* config() const { return _config; }

    // Access geometrical parameters

    /// Rotation angle of chamber (deg)
    inline float phiCh() const { return _PHICH; }

    /// Height of a cell (cm)
    inline float cellH() const { return _H; }

    /// width of a cell (cm) i.e. distance between ywo wires
    inline float cellPitch() const { return _PITCH; }

    /// distance between the phi view superlayers (cms)
    inline float distSL() const { return fabs(_ZSL[2]-_ZSL[0]); }

    /// coordinate of center of the 2 Phi SL
    inline float ZcenterSL() const { return 0.5*(_ZSL[2]+_ZSL[0]); } 

    /// radial coordinate in chamber frame of center of a superlayer
    float ZSL(int) const;

    /// number of BTIs in a required superlayer (i.e. nCells in lay 1)
    inline int nCell(int sl) const {
      return (sl>0&&sl<=3)*_NCELL[sl-1]; 
    }

    // NEWGEOmetry update
    /// staggering of first wire of layer respect to default: obsolete 19/6/06
    // int layerFEStaggering(int nsl, int nlay) const; 

    /// map tube number into hw wire number, and reverse hw num->tube 
    /// (nb NOT in bti hardware number, this depends on connectors)
    int mapTubeInFEch(int nsl, int nlay, int ntube) const; 

    /// sl offset in chamber front-end frame, in cm.
    float phiSLOffset();

    /// wire position in chamber frame
    LocalPoint tubePosInCh(int nsl, int nlay, int ntube) const;

    /// Front End position : 1=toward negative y, 0=toward positive y
    int posFE(int sl) const;


    // Local and global position of a trigger object
    /// Go to CMS coordinate system for a point
    GlobalPoint toGlobal(const LocalPoint p) const { return _stat->surface().toGlobal(p); }

    /// Go to CMS coordinate system for a vector
    GlobalVector toGlobal(const LocalVector v) const { return _stat->surface().toGlobal(v); }

    /// Go to Local coordinate system for a point
    LocalPoint toLocal(const GlobalPoint p) const { return _stat->surface().toLocal(p); }

    /// Go to Local coordinate system for a vector
    LocalVector toLocal(const GlobalVector v) const { return _stat->surface().toLocal(v); }

/*!
    \verbatim

    NB: attention: in NEWGEO definition has changed:

     +---------+---------+---------+
     | 1  o    | 5  o    | 9  o    |
     +----+----+----+----+----+----+
          | 3  o    |  7 o    |
     +----+----+----+----+----+ - - - -> x/-x
     | 2  o    | 6  o    |
     +----+----+----+----+----+
          | 4  o    | 8  o    |  
          +---------+---------+
          ^
          |
         x=0
    \endverbatim
  */
    /// Local position in chamber of a BTI
    LocalPoint localPosition(const DTBtiId) const;

 /*!
    \verbatim

    NB: attention: in NEWGEO definition has changed:

    +----+----+----+----+----+----+----+----+----+----+----+----+
    |  5 |  6 |  7 |  8 |  9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
    +----+----+----+----+----+----+----+----+----+----+----+----+
    \                                                           /
    | \                                                       /
    |   \                                                   /
    |     \                                               /
    |       \                                           /
----|-------------------------------------------------------------------> 
    |          \                                     /
    |            \                                 /
    |              \                             /
    |                \                         /
    |                  \                     /
    |                   +----+----+----+----+
    |                   |  1 |  2 |  3 |  4 |
    |                   +----+----+----+----+
    X=0
    ^
    |
   traco position

 
   \endverbatim
   */
    /// Local position in chamber of a TRACO
    LocalPoint localPosition(const DTTracoId) const;

    /// CMS position of a BTI
    inline GlobalPoint CMSPosition(const DTBtiId obj) const { 
      return  toGlobal(localPosition(obj));
    }

    /// CMS position of a TRACO
    inline GlobalPoint CMSPosition(const DTTracoId obj) const { 
      return toGlobal(localPosition(obj)); 
    }

    /// Dump the geometry
    void dumpGeom() const;

  private:

    /// get the geometry from the station
    void getGeom();

  private:

    DTChamber* _stat;     // Pointer to the chamber

    DTConfig* _config;    // Configuration parameters

    // geometrical parameters
    float _PHICH;       // angle of normal to the chamber in CMS frame (rad)
    float _H;           // height of a cell (cm)
    float _PITCH;       // width of a cell (cm)
    float _ZSL[3];      // Z coordinate of SL centers
    int _NCELL[3];      // number of cells (BTI) in SL each SL

};

#endif
