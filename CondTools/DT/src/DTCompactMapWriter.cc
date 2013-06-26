/** \class DTROSChannelId
 *
 *  See header file for a description of this class.
 *
 *  $Date: 2010/03/12 21:42:04 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//----------------------
// This Class' Header --
//----------------------
#include "CondTools/DT/interface/DTCompactMapWriter.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

 
//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <iostream>
#include <fstream>


class DTROSChannelId {
 public:
  DTROSChannelId( int ddu, int ros, int channel );
  int     dduId() const;
  int     rosId() const;
  int channelId() const;
 private:
  int     ddu_id;
  int     ros_id;
  int channel_id;
};

DTROSChannelId::DTROSChannelId( int ddu, int ros, int channel ):
     ddu_id(     ddu ),
     ros_id(     ros ),
 channel_id( channel ) {
}

int DTROSChannelId::dduId() const {
  return ddu_id;
}

int DTROSChannelId::rosId() const {
  return ros_id;
}

int DTROSChannelId::channelId() const {
  return channel_id;
}

class DTROSChannelCompare {
 public:
  bool operator()( const DTROSChannelId& idl,
                   const DTROSChannelId& idr ) const;
 private:
};

bool DTROSChannelCompare::operator()( const DTROSChannelId& idl,
                                       const DTROSChannelId& idr ) const {
  if ( idl.    dduId() < idr.    dduId() ) return true;
  if ( idl.    dduId() > idr.    dduId() ) return false;
  if ( idl.    rosId() < idr.    rosId() ) return true;
  if ( idl.    rosId() > idr.    rosId() ) return false;
  if ( idl.channelId() < idr.channelId() ) return true;
  if ( idl.channelId() > idr.channelId() ) return false;
  return false;
}


class DTROBCardId {
 public:
  DTROBCardId( int wheel, int sector, int station, int rob );
  int   wheelId() const;
  int  sectorId() const;
  int stationId() const;
  int     robId() const;
 private:
  int   wheel_id;
  int  sector_id;
  int station_id;
  int     rob_id;
};

DTROBCardId::DTROBCardId( int wheel, int sector, int station, int rob ):
   wheel_id(   wheel ),
  sector_id(  sector ),
 station_id( station ),
     rob_id(     rob ) {
}

int DTROBCardId::wheelId() const {
  return wheel_id;
}

int DTROBCardId::sectorId() const {
  return sector_id;
}

int DTROBCardId::stationId() const {
  return station_id;
}

int DTROBCardId::robId() const {
  return rob_id;
}


class DTROBCardCompare {
 public:
  bool operator()( const DTROBCardId& idl,
                   const DTROBCardId& idr ) const;
 private:
};

bool DTROBCardCompare::operator()( const DTROBCardId& idl,
                                    const DTROBCardId& idr ) const {
  if ( idl.  wheelId() < idr.  wheelId() ) return true;
  if ( idl.  wheelId() > idr.  wheelId() ) return false;
  if ( idl. sectorId() < idr. sectorId() ) return true;
  if ( idl. sectorId() > idr. sectorId() ) return false;
  if ( idl.stationId() < idr.stationId() ) return true;
  if ( idl.stationId() > idr.stationId() ) return false;
  if ( idl.    robId() < idr.    robId() ) return true;
  if ( idl.    robId() > idr.    robId() ) return false;
  return false;
}


class DTTDCChannelId {
 public:
  DTTDCChannelId( int tdc, int channel );
  int     tdcId() const;
  int channelId() const;
 private:
  int     tdc_id;
  int channel_id;
};

DTTDCChannelId::DTTDCChannelId( int tdc, int channel ):
     tdc_id(     tdc ),
 channel_id( channel ) {
}

int DTTDCChannelId::tdcId() const {
  return tdc_id;
}

int DTTDCChannelId::channelId() const {
  return channel_id;
}


class DTTDCChannelCompare {
 public:
  bool operator()( const DTTDCChannelId& idl,
                   const DTTDCChannelId& idr ) const;
 private:
};

bool DTTDCChannelCompare::operator()( const DTTDCChannelId& idl,
                                       const DTTDCChannelId& idr ) const {
  if ( idl.    tdcId() < idr.    tdcId() ) return true;
  if ( idl.    tdcId() > idr.    tdcId() ) return false;
  if ( idl.channelId() < idr.channelId() ) return true;
  if ( idl.channelId() > idr.channelId() ) return false;
  return false;
}


class DTPhysicalWireId {
 public:
  DTPhysicalWireId( int superlayer, int layer, int cell );
  int superlayerId() const;
  int      layerId() const;
  int       cellId() const;
 private:
  int superlayer_id;
  int      layer_id;
  int       cell_id;
};

DTPhysicalWireId::DTPhysicalWireId( int superlayer, int layer, int cell ):
 superlayer_id( superlayer ),
      layer_id(      layer ),
       cell_id(       cell ) {
}

int DTPhysicalWireId::superlayerId() const {
  return superlayer_id;
}

int DTPhysicalWireId::layerId() const {
  return layer_id;
}

int DTPhysicalWireId::cellId() const {
  return cell_id;
}


class DTPhysicalWireCompare {
 public:
  bool operator()( const DTPhysicalWireId& idl,
                   const DTPhysicalWireId& idr ) const;
 private:
};

bool DTPhysicalWireCompare::operator()( const DTPhysicalWireId& idl,
                                         const DTPhysicalWireId& idr ) const {
  if ( idl.superlayerId() < idr.superlayerId() ) return true;
  if ( idl.superlayerId() > idr.superlayerId() ) return false;
  if ( idl.     layerId() < idr.     layerId() ) return true;
  if ( idl.     layerId() > idr.     layerId() ) return false;
  if ( idl.      cellId() < idr.      cellId() ) return true;
  if ( idl.      cellId() > idr.      cellId() ) return false;
  return false;
}


class DTMapElementIdentifiers {
 public:
  static int defaultIdValue;
  static int ROSMapIdOffset;
  static int ROBMapIdOffset;
  static int TDCMapIdOffset;
};

int DTMapElementIdentifiers::defaultIdValue = 0x7fffffff;
int DTMapElementIdentifiers::ROSMapIdOffset = 2000001000;
int DTMapElementIdentifiers::ROBMapIdOffset = 2000002000;
int DTMapElementIdentifiers::TDCMapIdOffset = 2000003000;


void DTCompactMapWriter::buildSteering( std::istream& jobDesc ) {
  // template maps and lists, name and type
  std::string map_type( "dummy_string" );
  std::string map_file( "dummy_string" );
  std::string map_list( "dummy_string" );

  // ros-rob and tdc-cell maps
  std::map<DTROBCardId,
           int,
           DTROBCardCompare> tdc_idm;
  std::map<DTROSChannelId,DTROBCardId,DTROSChannelCompare> ros_map;
  std::map<DTROSChannelId,DTROBCardId,DTROSChannelCompare> rob_map;

  int rob_count = 1;
  int ros_count = 1;

  // parse job description file and loop over map templates and lists
  bool jobEnd = false;
  int max_line_length = 1000;
  char* job_line = new char[max_line_length];
  while ( jobDesc.getline( job_line, max_line_length ) ) {
    // parse job description line
    char* ptr_line = job_line;
    char* end_line = job_line + max_line_length;
    while( ptr_line < end_line ) {
      // get description command key
      // remove leading blanks
      while ( *ptr_line == ' ' ) ptr_line++;
      std::string key( ptr_line );
      int off_blank = key.find( " " );
      int off_equal = key.find( "=" );
      if ( off_equal < 0 ) off_equal = max_line_length;
      if ( off_blank < 0 ) off_blank = key.length();
      int ioff = ( off_blank < off_equal ? off_blank : off_equal );
      key.erase( ioff++, key.length() );
      // go to command value
      ptr_line += ++off_equal;
      // exit loop at end of description file
      if ( (jobEnd = ( key == "end" ) ) ) break;
      // get description command value
      while ( *ptr_line == ' ' ) ptr_line++;
      std::string val( ptr_line );
      int off_value = val.find( " " );
      if ( ( off_value > 0 ) &&
           ( off_value < static_cast<int>( val.length() ) ) )
      val.erase( off_value++, val.length() );
      // go to next token
      ptr_line += off_value;
      // set type, template or list file name
      if ( key == "type" ) map_type = val;
      if ( key ==  "map" ) map_file = val;
      if ( key == "list" ) map_list = val;
    }
    if ( jobEnd ) break;

    // open map list file
    std::ifstream lfile( map_list.c_str() );

    bool write;
    // store TDC maps
    if ( map_type == "TDC" ) {
      // loop over TDC map list for concerned wheels and sectors
      int station = 0;
      int  wheel;
      int sector;
      write = true;
      while ( lfile >>  wheel
                    >> sector
                    >> station ) fillTDCMap( map_file,
                                               wheel, sector,
                                             tdc_idm, station,
                                           rob_count, write );
      ++rob_count;
    }

    // store ROS maps
    if ( map_type == "ROS" ) {
      // loop over ROS map list for concerned ddu, ros, wheel and sector 
      int    ddu;
      int    ros;
      int  wheel;
      int sector;
      write = true;
      while ( lfile >> ddu
                    >> ros
                    >> wheel
                    >> sector ) fillROSMap( map_file,
                                                ddu,      ros,
                                              wheel,   sector,
                                            ros_map, rob_map,
                                          ros_count,    write );
      ++ros_count;
    }

  }
  delete job_line;

  // merge ROS and TDC maps
  fillReadOutMap( ros_count, tdc_idm, ros_map, rob_map );

  return;

}

void DTCompactMapWriter::fillTDCMap( const std::string& map_file,
                                      int wheel, int sector,
                                      std::map<DTROBCardId,
                                               int,
                                               DTROBCardCompare>& tdc_idm,
                                      int stationId,
                                      int map_count, bool& write ) {

  // template map file name
  std::ifstream ifile( map_file.c_str() );

  // get station number
  int station;
  int     rob;
  station = stationId;
  ifile >> rob;
  DTROBCardId key( wheel, sector, station, rob );
  tdc_idm.insert( std::pair<DTROBCardId,int>( key, map_count ) );
  if ( !write ) return;
  // loop over TDC channels and cells
  int superlayer;
  int      layer;
  int       wire;
  int        tdc;
  int    channel;
  while ( ifile >>        tdc
                >>    channel
                >> superlayer
                >>      layer
                >>       wire ) {
    // set ROB card, TDC channel and cell identifiers
    DTTDCChannelId tdcChannel( tdc, channel );
    DTPhysicalWireId physicalWire( superlayer, layer, wire );
    // look for rob card connection map
    std::cout << DTMapElementIdentifiers::TDCMapIdOffset << " "
              <<                               map_count << " "
              <<                                     rob << " "
              <<                                     tdc << " "
              <<                                 channel << " "
              << DTMapElementIdentifiers::defaultIdValue << " "
              << DTMapElementIdentifiers::defaultIdValue << " "
              << DTMapElementIdentifiers::defaultIdValue << " "
              <<                              superlayer << " "
              <<                                   layer << " "
              <<                                    wire
              << std::endl;

  }
  write = false;
  return;
}

void DTCompactMapWriter::fillROSMap( const std::string& map_file,
                                      int   ddu, int    ros,
                                      int whdef, int secdef,
                                      std::map<DTROSChannelId,
                                               DTROBCardId,
                                               DTROSChannelCompare>& ddu_map,
                                      std::map<DTROSChannelId,
                                               DTROBCardId,
                                               DTROSChannelCompare>& ros_map,
                                      int map_count,
                                      bool& write ) {

  DTROSChannelId rosBoard( ddu, ros, 0 );
  DTROBCardId    sectorId( whdef, secdef, 0, map_count );
  ddu_map.insert( std::pair<DTROSChannelId,DTROBCardId>( rosBoard,
                                                           sectorId ) );

  // template map file name
  std::ifstream ifile( map_file.c_str() );

  // loop over ROS channels and ROBs
  int channel;
  int wheel;
  int sector;
  int station;
  int rob;
  DTROSChannelId rosChanMap( ddu, ros, 0 );
  if ( !write ) return;
  while ( ifile >> channel
                >>   wheel
                >>  sector
                >> station
                >>     rob ) {
    // set sector to default unless specified
    if ( !sector ) sector = DTMapElementIdentifiers::defaultIdValue;
    if ( ! wheel )  wheel = DTMapElementIdentifiers::defaultIdValue;
    // set ROS channel and ROB identifiers
    DTROSChannelId rosChannel( ddu, ros, channel );
    DTROBCardId robCard( wheel, sector, station, rob );
    // store ROS channel to ROB connection into map
    ros_map.insert( std::pair<DTROSChannelId,DTROBCardId>( rosChannel,
                                                             robCard ) );
  }
  write = false;

  return;
}

void DTCompactMapWriter::cloneROS(  std::map<DTROSChannelId,
                                              DTROBCardId,
                                              DTROSChannelCompare>& ros_m,
                                     std::map<DTROSChannelId,
                                              DTROBCardId,
                                              DTROSChannelCompare>& ros_a,
                                     int ddu_o, int ros_o,
                                     int ddu_f, int ros_f ) {

  std::map<DTROSChannelId,
           DTROBCardId,
           DTROSChannelCompare>:: const_iterator ros_iter = ros_m.begin();
  std::map<DTROSChannelId,
           DTROBCardId,
           DTROSChannelCompare>:: const_iterator ros_iend = ros_m.end();
  while ( ros_iter != ros_iend ) {
    const std::pair<DTROSChannelId,DTROBCardId>& rosLink = *ros_iter++;
    const DTROSChannelId& rosChannelId = rosLink.first;
    const DTROBCardId&    robCardId    = rosLink.second;
    if ( rosChannelId.dduId() != ddu_o ) continue;
    if ( rosChannelId.rosId() != ros_o ) continue;
    const DTROSChannelId  rosChanNewId( ddu_f, ros_f,
                                         rosChannelId.channelId() );
    ros_a.insert( std::pair<DTROSChannelId,DTROBCardId>( rosChanNewId,
                                                           robCardId ) );
  }
  return;  
}

void DTCompactMapWriter::appendROS( std::map<DTROSChannelId,
                                              DTROBCardId,
                                              DTROSChannelCompare>& ros_m,
                                     std::map<DTROSChannelId,
                                              DTROBCardId,
                                              DTROSChannelCompare>& ros_a ) {

  std::map<DTROSChannelId,
           DTROBCardId,
           DTROSChannelCompare>:: const_iterator ros_iter = ros_a.begin();
  std::map<DTROSChannelId,
           DTROBCardId,
           DTROSChannelCompare>:: const_iterator ros_iend = ros_a.end();
  while ( ros_iter != ros_iend ) ros_m.insert( *ros_iter++ );
  return;  
}


void DTCompactMapWriter::fillReadOutMap(
                         int ros_count,
                         std::map<DTROBCardId,
                                  int,
                                  DTROBCardCompare>& tdc_idm,
                         std::map<DTROSChannelId,
                                  DTROBCardId,
                                  DTROSChannelCompare>& ddu_map,
                         std::map<DTROSChannelId,
                                  DTROBCardId,
                                  DTROSChannelCompare>& ros_map ) {

  bool check = true;
  bool write = false;

  while ( check || write ) {
    // ROS board to sector connection map
    std::map<DTROSChannelId,
             DTROBCardId,
             DTROSChannelCompare>:: const_iterator ddu_iter = ddu_map.begin();
    std::map<DTROSChannelId,
             DTROBCardId,
             DTROSChannelCompare>:: const_iterator ddu_iend = ddu_map.end();
    // ROB card to TDC element map
    std::map<DTROBCardId,
             int,
             DTROBCardCompare>::const_iterator idt_iend = tdc_idm.end();
    // ROS channel to ROB connection map
    std::map<DTROSChannelId,
             DTROBCardId,
             DTROSChannelCompare>:: const_iterator ros_iter = ros_map.begin();
    std::map<DTROSChannelId,
             DTROBCardId,
             DTROSChannelCompare>:: const_iterator ros_iend = ros_map.end();

    check = false;
    std::map<DTROSChannelId,
             DTROBCardId,
             DTROSChannelCompare> ros_app;

    // loop over ROS channels
    while ( ros_iter != ros_iend ) {
      // get ROS channel and ROB identifiers
      const std::pair<DTROSChannelId,DTROBCardId>& rosLink = *ros_iter++;
      const DTROSChannelId& rosChannelId = rosLink.first;
      const DTROSChannelId  rosChanMapId( rosChannelId.dduId(),
                                           rosChannelId.rosId(), 0 );
      const DTROBCardId&    robCMapId    = rosLink.second;
      ddu_iter = ddu_map.find( rosChanMapId );
      int  whdef = 0;
      int secdef = 0;
      int rosMapCk = 0;
      if ( ddu_iter != ddu_iend ) {
        const DTROBCardId& sector = ddu_iter->second;
         whdef = sector.wheelId();
        secdef = sector.sectorId();
        rosMapCk = sector.robId();
      }
      else {
        std::cout << "DDU map "
                  << rosChannelId.dduId() << " "
                  << rosChannelId.rosId() << " 0 not found" << std::endl;
        continue;
      }
      int wheel   = robCMapId.  wheelId();
      int sector  = robCMapId. sectorId();
      int station = robCMapId.stationId();
      int rob     = robCMapId.    robId();
      if (  wheel != DTMapElementIdentifiers::defaultIdValue )  whdef =  wheel;
      if ( sector != DTMapElementIdentifiers::defaultIdValue ) secdef = sector;
      const DTROBCardId robCardId( whdef, secdef, station, rob );
      std::map<DTROBCardId,
               int,
               DTROBCardCompare>::const_iterator idt_iter =
                                   tdc_idm.find( robCardId );
      if ( idt_iter != idt_iend ) {
        int tdcMapId = idt_iter->second;
        ddu_iter = ddu_map.begin();
        while ( ddu_iter != ddu_iend ) {
          const std::pair<DTROSChannelId,DTROBCardId>& dduLink = *ddu_iter++;
          const DTROSChannelId& ros_id = dduLink.first;
          const DTROBCardId&    sec_id = dduLink.second;
          int whe_chk = sec_id.wheelId();
          int sec_chk = sec_id.sectorId();
          if (  wheel != DTMapElementIdentifiers::defaultIdValue )
               whe_chk =  wheel;
          if ( sector != DTMapElementIdentifiers::defaultIdValue )
               sec_chk = sector;
          if ( rosMapCk != sec_id.robId() ) continue;
          DTROBCardId robCardId( whe_chk, sec_chk,
                                  station, rob );
          idt_iter = tdc_idm.find( robCardId );
          int tdcMapCk = idt_iter->second;
          if ( ( tdcMapId != tdcMapCk ) && ( !check ) ) {
            cloneROS( ros_map, ros_app, 
                      rosChannelId.dduId(), rosChannelId.rosId(),
                      ros_id.      dduId(), ros_id.      rosId() );
            tdcMapId = tdcMapCk;
            check = true;
          }
          if ( ( tdcMapId == tdcMapCk ) && check ) {
            std::map<DTROSChannelId,
                     DTROBCardId,
                     DTROSChannelCompare>::iterator ddu_inew =
                                            ddu_map.find( ros_id );
            ddu_inew->second = DTROBCardId( sec_id.wheelId(),
                                             sec_id.sectorId(),
                                             sec_id.stationId(),
                                             ros_count );
          }
        }
        if ( check ) {
          ros_count++;
          break;
        }
        if ( write )
          std::cout << DTMapElementIdentifiers::ROSMapIdOffset << " "
                    <<                                rosMapCk << " "
                    <<                rosChannelId.channelId() << " "
                    << DTMapElementIdentifiers::defaultIdValue << " "
                    << DTMapElementIdentifiers::defaultIdValue << " "
                    <<                                   wheel << " "
                    <<                                 station << " "
                    <<                                  sector << " "
                    <<                                     rob << " "
                    << DTMapElementIdentifiers::TDCMapIdOffset << " "
                    <<                                tdcMapId << std::endl;

      }
      else {
        std::cout << "TDC map "
                  <<   wheel << " "
                  <<  sector << " "
                  << station << " "
                  <<     rob << " not found" << std::endl;
      }
    }
    appendROS( ros_map, ros_app );
    if ( !write ) {
      write = !check;
    }
    else {
      ddu_iter = ddu_map.begin();
      while ( ddu_iter != ddu_iend ) {
        const std::pair<DTROSChannelId,DTROBCardId>& dduLink = *ddu_iter++;
        const DTROSChannelId& ros_id = dduLink.first;
        const DTROBCardId&    sec_id = dduLink.second;
        std::cout <<                          ros_id.dduId() << " "
                  <<                          ros_id.rosId() << " "
                  << DTMapElementIdentifiers::defaultIdValue << " "
                  << DTMapElementIdentifiers::defaultIdValue << " "
                  << DTMapElementIdentifiers::defaultIdValue << " "
                  <<                        sec_id.wheelId() << " "
                  << DTMapElementIdentifiers::defaultIdValue << " "
                  <<                       sec_id.sectorId() << " "
                  << DTMapElementIdentifiers::defaultIdValue << " "
                  << DTMapElementIdentifiers::ROSMapIdOffset << " "
                  <<                          sec_id.robId() << std::endl;
      }
      write = false;
    }
  }

  return;
}

