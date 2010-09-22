#ifndef Fireworks_DetIdInfo_h
#define Fireworks_DetIdInfo_h

/**
 * 
 *  Description:
 *    Helper class to get human readable informationa about given DetId
 *    (copied from TrackingTools/TrackAssociator)
 * 
 */

#include "DataFormats/DetId/interface/DetId.h"
#include <set>
#include <vector>

class FWDetIdInfo {
 public:
   static std::string info( const DetId& );
   static std::string info( const std::set<DetId>& );
   static std::string info( const std::vector<DetId>& );
};
#endif
