#ifndef DTCompactMapWriter_H
#define DTCompactMapWriter_H
/** \class DTCompactMapWriter
 *
 *  Description:
 *       Class to build readout maps from map templates and lists
 *
 *  $Date: 2009/03/26 14:11:03 $
 *  $Revision: 1.1 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// Base Class Headers --
//----------------------


//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
class DTROBCardId;
class DTROBCardCompare;
class DTTDCChannelId;
class DTTDCChannelCompare;
class DTPhysicalWireId;
class DTPhysicalWireCompare;
class DTROSChannelId;
class DTROSChannelCompare;

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <iostream>
#include <fstream>


class DTCompactMapWriter {
 public:
  static void buildSteering( std::istream& jobDesc );
 private:
  static void fillTDCMap( const std::string& map_file,
                          int wheel, int sector,
                          std::map<DTROBCardId,
                                   int,
                                   DTROBCardCompare>& tdc_idm,
                          int stationId, // int robId,
                          int map_count, bool& write );
  static void fillROSMap( const std::string& map_file,
                          int   ddu, int    ros,
                          int whdef, int secdef,
                          std::map<DTROSChannelId,
                                   DTROBCardId,
                                   DTROSChannelCompare>& ddu_map,
                          std::map<DTROSChannelId,
                                   DTROBCardId,
                                   DTROSChannelCompare>& ros_map,
                          int map_count,
                          bool& write );
  static void cloneROS(   std::map<DTROSChannelId,
                                   DTROBCardId,
                                   DTROSChannelCompare>& ros_m,
                          std::map<DTROSChannelId,
                                   DTROBCardId,
                                   DTROSChannelCompare>& ros_a,
                          int ddu_o, int ros_o,
                          int ddu_f, int ros_f );
  static void appendROS(  std::map<DTROSChannelId,
                                   DTROBCardId,
                                   DTROSChannelCompare>& ros_m,
                          std::map<DTROSChannelId,
                                   DTROBCardId,
                                   DTROSChannelCompare>& ros_a );
  static void fillReadOutMap( int ros_count,
                              std::map<DTROBCardId,
                                       int,
                                       DTROBCardCompare>& tdc_idm,
                              std::map<DTROSChannelId,
                                       DTROBCardId,
                                       DTROSChannelCompare>& ddu_map,
                              std::map<DTROSChannelId,
                                       DTROBCardId,
                                       DTROSChannelCompare>& ros_map );
};

#endif

