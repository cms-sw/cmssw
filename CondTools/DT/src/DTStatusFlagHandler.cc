/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/09/25 12:03:21 $
 *  $Revision: 1.4 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/interface/DTStatusFlagHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTStatusFlag.h"

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
DTStatusFlagHandler::DTStatusFlagHandler( const edm::ParameterSet& ps ):
 dataTag(   ps.getParameter<std::string>  (  "tag" ) ),
 fileName(  ps.getParameter<std::string>  ( "file" ) ),
 runNumber( ps.getParameter<unsigned int> (  "run" ) ) {
}

//--------------
// Destructor --
//--------------
DTStatusFlagHandler::~DTStatusFlagHandler() {
}

//--------------
// Operations --
//--------------
void DTStatusFlagHandler::getNewObjects() {

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

  DTStatusFlag* stFlag = new DTStatusFlag( dataTag );

  int status = 0;
  std::ifstream ifile( fileName.c_str() );
  int whe;
  int sta;
  int sec;
  int qua;
  int lay;
  int cel;
  int noiseFlag;
  int    feMask;
  int   tdcMask;
  int  trigMask;
  int  deadFlag;
  int  nohvFlag;
  while ( ifile >> whe
                >> sta
                >> sec
                >> qua
                >> lay
                >> cel
                >> noiseFlag
                >> feMask
                >> tdcMask
                >> trigMask
                >> deadFlag
                >> nohvFlag ) {
    status = stFlag->set( whe, sta, sec, qua, lay, cel,
                          noiseFlag != 0,
                             feMask != 0,
                            tdcMask != 0,
                           trigMask != 0,
                           deadFlag != 0,
                           nohvFlag != 0 );
    std::cout << whe << " "
              << sta << " "
              << sec << " "
              << qua << " "
              << lay << " "
              << cel << " "
              << noiseFlag << " "
              <<    feMask << " "
              <<   tdcMask << " "
              <<  trigMask << " "
              <<  deadFlag << " "
              <<  nohvFlag << "  -> ";                
    std::cout << "insert status: " << status << std::endl;
  }

/*
  unsigned int runf = irun;
  unsigned int runl = 0xffffffff;
  popcon::IOVPair iop = { runf, runl };
  std::cout << "APPEND NEW OBJECT: "
            << runf << " " << runl << " " << stFlag << std::endl;
  m_to_transfer->push_back( std::make_pair( stFlag, iop ) );
*/

  //for each payload provide IOV information (say in this case we use since)
  cond::Time_t snc = runNumber;
  if ( runNumber > last )
       m_to_transfer.push_back( std::make_pair( stFlag, snc ) );
  else
       std::cout << "More recent data already present - skipped" << std::endl;

  return;

}


std::string DTStatusFlagHandler::id() const {
  return dataTag;
}


