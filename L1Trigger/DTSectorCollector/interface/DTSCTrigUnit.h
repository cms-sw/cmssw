//-------------------------------------------------
//
/**  \class  DTSCTrigUnit
 *     Muon Barrel Sector Collector Trigger Unit (Chamber trigger)
 *
 *   $Date: 2010/11/11 16:28:21 $
 *   $Revision: 1.10 $
 *
 *   \author C.Grandi, S. Marcellini
 */
//
//--------------------------------------------------
#ifndef DT_SC_TRIG_UNIT_H
#define DT_SC_TRIG_UNIT_H

//---------------
// C++ Headers --
//---------------
#include <vector>

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfigManager.h"
#include "L1Trigger/DTUtilities/interface/DTTrigGeom.h"
#include "L1Trigger/DTUtilities/interface/DTTrigData.h"
#include "L1Trigger/DTBti/interface/DTBtiChip.h"
#include "L1Trigger/DTTraco/interface/DTTracoChip.h"
#include "L1Trigger/DTTriggerServerPhi/interface/DTChambPhSegm.h"
#include "L1Trigger/DTTriggerServerTheta/interface/DTChambThSegm.h"
#include "L1Trigger/DTUtilities/interface/DTGeomSupplier.h"
#include "L1Trigger/DTBti/interface/DTBtiCard.h"
#include "L1Trigger/DTTraco/interface/DTTracoCard.h"
#include "L1Trigger/DTTriggerServerPhi/interface/DTTSPhi.h"
#include "L1Trigger/DTTriggerServerTheta/interface/DTTSTheta.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/LocalVector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class DTChamber;


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTSCTrigUnit {

  public:

    /// Constructor
    //DTSCTrigUnit(DTChamber* stat, edm::ParameterSet& tu_pset) ;
    DTSCTrigUnit(DTChamber *stat) ;  

    /// Destructor 
    ~DTSCTrigUnit() ;
 
    /// The associated geometry
    inline DTTrigGeom* geom() const { return _geom; }

    /// Set geometry
    void setGeom(const DTChamber* stat) { _geom->setGeom(stat); }

    /// Set configuration
    void setConfig(const DTConfigManager *conf);

    /// The associated chamber
    inline const DTChamber* stat() const { return _geom->stat(); }

    /// Identifier of the associated chamber
    inline DTChamberId statId() const { return _geom->statId(); }

    /// Return wheel number
    inline int wheel() const { return _geom->wheel(); }

    /// Return station number
    inline int station() const { return _geom->station(); }

    /// Return sector number
    inline int sector() const { return _geom->sector(); }

    /// Return container of BTI triggers
    inline DTBtiCard* BtiTrigs() const { return _theBTIs; }

    /// Return container of TRACO triggers
    inline DTTracoCard* TracoTrigs() const { return  _theTRACOs; }

    /// Return the chamber Trigger Server (Phi)
    inline DTTSPhi* TSPhTrigs() const { return _theTSPhi; }

    /// Return the chamber Trigger Server (Theta)
    inline DTTSTheta* TSThTrigs() const { return _theTSTheta; }

    /// Return the appropriate coordinate supplier
    DTGeomSupplier* GeomSupplier(const DTTrigData* trig) const;

    /// Coordinate of a trigger-data object in chamber frame
    inline LocalPoint localPosition(const DTTrigData* trig) const {
      return GeomSupplier(trig)->localPosition(trig);
    }

    /// Coordinate of a trigger-data object in CMS frame
    inline GlobalPoint CMSPosition(const DTTrigData* trig) const {
      return GeomSupplier(trig)->CMSPosition(trig);
    }

    /// Direction of a trigger-data object in chamber frame
    inline LocalVector localDirection(const DTTrigData* trig) const {
      return GeomSupplier(trig)->localDirection(trig);
    }

    /// Direction of a trigger-data object in CMS frame
    inline GlobalVector CMSDirection(const DTTrigData* trig) const {
      return GeomSupplier(trig)->CMSDirection(trig);
    }

    /// Print a trigger-data object 
    inline void print(DTTrigData* trig) const {
      GeomSupplier(trig)->print(trig);
    }

    /// Dump the geometry
    inline void dumpGeom() const { _geom->dumpGeom(); }

    /// Dump the Lut file
    inline void dumpLUT(short int btic) const { _geom->dumpLUT(btic); }

    /// Number of active DTBtiChips
    int nDTBtiChip() { return _theBTIs->size(); }

    /// Number of active DTTracoChips
    int nDTTracoChip() { return _theTRACOs->size(); }

    /// Number of Phi segments for a given step
    int nPhiSegm(int step) { return _theTSPhi->nSegm(step); }


    /// Return output segments, phi view
    const DTChambPhSegm* phiSegment(int step, int n) { 
      return _theTSPhi->segment(step, n); 
    }

    /// Number of theta segments for a given step
    int nThetaSegm(int step) { return _theTSTheta->nSegm(step); }

    /// Return output segments, theta view
    const DTChambThSegm* thetaSegment(int step, int n) { 
      return _theTSTheta->segment(step, n); 
    }

  private:

    DTTrigGeom* _geom;         		// Pointer to the geometry
  
    // Containers for DTBtiChip, DTTracoChip and TS 
    DTBtiCard* _theBTIs;
    DTTracoCard* _theTRACOs;
    DTTSPhi* _theTSPhi;
    DTTSTheta* _theTSTheta;

};

#endif
