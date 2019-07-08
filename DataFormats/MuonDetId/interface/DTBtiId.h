//-------------------------------------------------
//
/**  \class DTBtiId
 *    BTI Identifier
 *
 *   \author C.Grandi
 */
//
//--------------------------------------------------
#ifndef DT_BTI_ID_H_
#define DT_BTI_ID_H_

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//----------------------
// Base Class Headers --
//----------------------
#include "DataFormats/MuonDetId/interface/DTChamberId.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"

//---------------
// C++ Headers --
//---------------

//              ---------------------
//              -- Class Interface --
//              ---------------------

//typedef unsigned char myint8;
class DTBtiId {
public:
  ///  Constructor
  DTBtiId() : _bti(0) {}

  ///  Constructor
  DTBtiId(const DTSuperLayerId& mu_superlayer_id, const int bti_id) : _suplId(mu_superlayer_id), _bti(bti_id) {}

  ///  Constructor
  DTBtiId(const DTChamberId& mu_stat_id, const int superlayer_id, const int bti_id)
      : _suplId(mu_stat_id, superlayer_id), _bti(bti_id) {}

  ///  Constructor
  DTBtiId(const int wheel_id, const int station_id, const int sector_id, const int superlayer_id, const int bti_id)
      : _suplId(wheel_id, station_id, sector_id, superlayer_id), _bti(bti_id) {}

  ///  Constructor
  DTBtiId(const DTBtiId& btiId) : _suplId(btiId._suplId), _bti(btiId._bti) {}

  /// Destructor
  virtual ~DTBtiId() {}

  /// Returns wheel number
  inline int wheel() const { return _suplId.wheel(); }
  /// Returns station number
  inline int station() const { return _suplId.station(); }
  /// Returns sector number
  inline int sector() const { return _suplId.sector(); }
  /// Returns the superlayer
  inline int superlayer() const { return _suplId.superlayer(); }
  /// Returns the bti
  inline int bti() const { return _bti; }
  /// Returns the superlayer id
  inline DTSuperLayerId SLId() const { return _suplId; }

  bool operator==(const DTBtiId& id) const {
    if (wheel() != id.wheel())
      return false;
    if (sector() != id.sector())
      return false;
    if (station() != id.station())
      return false;
    if (superlayer() != id.superlayer())
      return false;
    if (_bti != id.bti())
      return false;
    return true;
  }

  bool operator<(const DTBtiId& id) const {
    if (wheel() < id.wheel())
      return true;
    if (wheel() > id.wheel())
      return false;

    if (station() < id.station())
      return true;
    if (station() > id.station())
      return false;

    if (sector() < id.sector())
      return true;
    if (sector() > id.sector())
      return false;

    if (superlayer() < id.superlayer())
      return true;
    if (superlayer() > id.superlayer())
      return false;

    if (bti() < id.bti())
      return true;
    if (bti() > id.bti())
      return false;

    return false;
  }

private:
  DTSuperLayerId _suplId;  // this is 4 bytes
  int _bti;
};

#endif
