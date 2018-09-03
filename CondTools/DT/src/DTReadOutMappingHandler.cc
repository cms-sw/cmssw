/*
 *  See header file for a description of this class.
 *
 *  $Date: 2007/12/07 15:13:50 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/interface/DTReadOutMappingHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTReadOutMapping.h"

//---------------
// C++ Headers --
//---------------
#include <iostream>
#include <fstream>

//-------------------
// Initializations --
//-------------------


//----------------
// Constructors --
//----------------
DTReadOutMappingHandler::DTReadOutMappingHandler( const edm::ParameterSet& ps ):
 dataTag(   ps.getParameter<std::string>  (  "tag" ) ),
 fileName(  ps.getParameter<std::string>  ( "file" ) ),
 runNumber( ps.getParameter<unsigned int> (  "run" ) ) {
}

//--------------
// Destructor --
//--------------
DTReadOutMappingHandler::~DTReadOutMappingHandler() {
}

//--------------
// Operations --
//--------------
void DTReadOutMappingHandler::getNewObjects() {

  //to access the information on the tag inside the offline database:
  cond::TagInfo const & ti = tagInfo();
  unsigned int last = ti.lastInterval.first;

  //to access the information on last successful log entry for this tag:
//  cond::LogDBEntry const & lde = logDBEntry();     

  //to access the lastest payload (Ref is a smart pointer)
//  Ref payload = lastPayload();

/*
  int irun = event.id().run();
  int ievt = event.id().event();
  std::cout << "================ "
            << irun << " " << ievt << std::endl;

  std::map<std::string, popcon::PayloadIOV> mp = getOfflineInfo();
  std::map<std::string, popcon::PayloadIOV>::iterator iter = mp.begin();
  std::map<std::string, popcon::PayloadIOV>::iterator iend = mp.end();
  std::cout << "list of all tags: " << std::endl;
  while ( iter != iend ) {
    std::cout << "Tag: "                       << iter->first
              << " , last object valid since " << iter->second.last_since
              << " to "                        << iter->second.last_till
              << std::endl;
    iter++;
  }

  std::cout << "look for tag " << dataTag << std::endl;
  std::map<std::string, popcon::PayloadIOV>::iterator itag =
    mp.find( dataTag );
*/

  std::string robMap( dataTag );
  std::string rosMap( dataTag );
  robMap += "_ROB";
  rosMap += "_ROS";
  DTReadOutMapping* ro_map = new DTReadOutMapping( robMap, rosMap );

  int status = 0;
  std::ifstream ifile( fileName.c_str() );
  int ddu;
  int ros;
  int rob;
  int tdc;
  int cha;
  int whe;
  int sta;
  int sec;
  int qua;
  int lay;
  int cel;
  while ( ifile >> ddu
                >> ros
                >> rob
                >> tdc
                >> cha
                >> whe
                >> sta
                >> sec
                >> qua
                >> lay
                >> cel ) {
    status = ro_map->insertReadOutGeometryLink( ddu, ros, rob, tdc, cha,
                                                whe, sta, sec,
                                                qua, lay, cel );
    std::cout << ddu << " "
              << ros << " "
              << rob << " "
              << tdc << " "
              << cha << " "
              << whe << " "
              << sta << " "
              << sec << " "
              << qua << " "
              << lay << " "
              << cel << "  -> ";                
    std::cout << "insert status: " << status << std::endl;
  }

/*
  unsigned int runf = irun;
  unsigned int runl = 0xffffffff;
  popcon::IOVPair iop = { runf, runl };
  std::cout << "APPEND NEW OBJECT: "
            << runf << " " << runl << " " << ro_map << std::endl;
  m_to_transfer->push_back( std::make_pair( ro_map, iop ) );
*/

  //for each payload provide IOV information (say in this case we use since)
  cond::Time_t snc = runNumber;
  if ( runNumber > last )
       m_to_transfer.push_back( std::make_pair( ro_map, snc ) );
  else {
       std::cout << "More recent data already present - skipped" << std::endl;
       delete ro_map;
  }

  return;

}


std::string DTReadOutMappingHandler::id() const {
  return dataTag;
}


