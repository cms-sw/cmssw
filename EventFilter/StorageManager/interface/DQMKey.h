// $Id$

#ifndef StorageManager_DQMKey_h
#define StorageManager_DQMKey_h

#include <cstddef>

namespace stor {

  /**
   * Definition of the DQMKey used in the storage manager
   *
   * $Author$
   * $Revision$
   * $Date$
   */
  
  struct DQMKey
  {
    uint32_t runNumber;
    uint32_t lumiSection;
    uint32_t updateNumber;

    bool operator<(DQMKey const& other) const;
    bool operator==(DQMKey const& other) const;
  };
  
  inline bool DQMKey::operator<(DQMKey const& other) const
  {
    if ( runNumber != other.runNumber) return runNumber < other.runNumber;
    if ( lumiSection != other.lumiSection) return lumiSection < other.lumiSection;
    return updateNumber < other.updateNumber;
  }
  
  inline bool DQMKey::operator==(DQMKey const& other) const
  {
    return (runNumber == other.runNumber &&
      lumiSection == other.lumiSection &&
      updateNumber == other.updateNumber);
  }

} // namespace stor

#endif // StorageManager_DQMKey_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
