//-------------------------------------------------
//
/**  \class DTTrigData
 *     Defines the ability to calculate coordinates of Level1 MuDT Trigger objects
 *
 *   \author C.Grandi
 */
//
//--------------------------------------------------
#ifndef DT_TRIG_DATA_H_
#define DT_TRIG_DATA_H_

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//----------------------
// Base Class Headers --
//----------------------
//#include "Profound/MuNumbering/interface/MuBarIdInclude.h"
//Should become
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTTrigData {

 public:

  ///  Constructor
  DTTrigData() {}

   /// Destructor
  virtual ~DTTrigData() {}

  /// Identifier of the associated chamber
  virtual DTChamberId ChamberId() const = 0;

  /// Return wheel number
  inline int wheel() const { return ChamberId().wheel(); }

  /// Return station number
  inline int station() const { return ChamberId().station(); }

  /// Return sector number
  inline int sector() const { return ChamberId().sector(); }

  /// Print a trigger-data object with also local and global position/direction
  virtual void print() const = 0;

};
#endif




