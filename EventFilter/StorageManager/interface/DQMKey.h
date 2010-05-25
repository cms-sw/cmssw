// $Id: DQMKey.h,v 1.4 2009/10/13 15:08:33 mommsen Exp $
/// @file: DQMKey.h 

#ifndef StorageManager_DQMKey_h
#define StorageManager_DQMKey_h

#include <cstddef>
#include <stdint.h>

namespace stor {

  /**
   * Definition of the DQMKey used in the storage manager
   *
   * $Author: mommsen $
   * $Revision: 1.4 $
   * $Date: 2009/10/13 15:08:33 $
   */
  
  struct DQMKey
  {
    uint32_t runNumber;
    uint32_t lumiSection;

    bool operator<(DQMKey const& other) const;
    bool operator==(DQMKey const& other) const;
  };
  
  inline bool DQMKey::operator<(DQMKey const& other) const
  {
    if ( runNumber != other.runNumber) return runNumber < other.runNumber;
    return lumiSection < other.lumiSection;
  }
  
  inline bool DQMKey::operator==(DQMKey const& other) const
  {
    return (runNumber == other.runNumber &&
      lumiSection == other.lumiSection);
  }

} // namespace stor

#endif // StorageManager_DQMKey_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
