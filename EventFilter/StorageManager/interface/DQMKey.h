// $Id: DQMKey.h,v 1.6 2011/03/07 15:31:31 mommsen Exp $
/// @file: DQMKey.h 

#ifndef EventFilter_StorageManager_DQMKey_h
#define EventFilter_StorageManager_DQMKey_h

#include <cstddef>
#include <stdint.h>
#include <string>

namespace stor {

  /**
   * Definition of the DQMKey used in the storage manager
   *
   * $Author: mommsen $
   * $Revision: 1.6 $
   * $Date: 2011/03/07 15:31:31 $
   */
  
  struct DQMKey
  {
    uint32_t runNumber;
    uint32_t lumiSection;
    std::string topLevelFolderName;

    bool operator<(DQMKey const& other) const;
    bool operator==(DQMKey const& other) const;
  };
  
  inline bool DQMKey::operator<(DQMKey const& other) const
  {
    if ( runNumber != other.runNumber ) return runNumber < other.runNumber;
    if ( lumiSection != other.lumiSection ) return lumiSection < other.lumiSection;
    return ( topLevelFolderName < other.topLevelFolderName );
  }
  
  inline bool DQMKey::operator==(DQMKey const& other) const
  {
    return ( runNumber == other.runNumber &&
      lumiSection == other.lumiSection &&
      topLevelFolderName == other.topLevelFolderName );
  }

} // namespace stor

#endif // EventFilter_StorageManager_DQMKey_h 


/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
