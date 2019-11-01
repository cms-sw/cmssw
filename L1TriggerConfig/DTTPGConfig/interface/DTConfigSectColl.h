//-------------------------------------------------
//
/**  \class DTConfigSectColl
 *
 *   Configurable parameters and constants 
 *   for Level-1 Muon DT Trigger - SectorCollector
 *
 *
 *   \author c. Battilana
 *
 */
//
//--------------------------------------------------
#ifndef DT_CONFIG_SECTCOLL_H
#define DT_CONFIG_SECTCOLL_H

//---------------
// C++ Headers --
//---------------
#include <iostream>

//----------------------
// Base Class Headers --
//----------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "L1TriggerConfig/DTTPGConfig/interface/DTConfig.h"

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTConfigSectColl : public DTConfig {
public:
  //! Constants: number of TSTheta/TSPhi in input to Sector Collector
  static const int NTSTSC = 3, NTSPSC = 5;

  //! Constant: maximum number of Sector Collector sorting Chip in input to Sector Collector
  static const int NDTSC = 4;

  //! Constructor
  DTConfigSectColl(const edm::ParameterSet& ps);

  //! Constructor
  DTConfigSectColl();

  //! Destructor
  ~DTConfigSectColl() override;

  //! Return the debug flag
  inline bool debug() const { return m_debug; }

  //! Return carry in Sector Collector for station istat (1 means enabled, 0 disabled)
  inline bool SCGetCarryFlag(int istat) const {
    if (istat < 1 || istat > 4) {
      throw cms::Exception("DTTPG") << "DTConfigSectColl::SCGetCarryFlag: station number out of range: istat=" << istat
                                    << std::endl;
    }
    return m_scecf[istat - 1];
  }

  //! Return coarsesync parameter in Sector Collector for station istat (5 is second MB4 station)
  inline int CoarseSync(int istat) const {
    if (istat < 1 || istat > 5) {
      throw cms::Exception("DTTPG") << "DTConfigSectColl::CoarseSync: station number out of range: istat=" << istat
                                    << std::endl;
    }
    return m_sccsp[istat - 1];
  }

  // Set Methods
  //! Set debug flag
  inline void setDebug(bool debug) { m_debug = debug; }

  //! Set carry in Sector Collector for station istat (1 means enabled, 0 disabled)
  void setSCCarryFlag(bool scecf, int istat);

  //! Return coarsesync parameter in Sector Collector for station istat (5 is second MB4 station)
  void setCoarseSync(int sccsp, int istat);

  //! Print the setup
  void print() const;

private:
  //! Load pset values into class variables
  void setDefaults(const edm::ParameterSet& ps);

  bool m_debug;
  bool m_scecf[4];
  int m_sccsp[5];
};

#endif
