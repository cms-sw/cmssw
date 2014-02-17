/*
 *  See header file for a description of this class.
 *
 *  $Date: 2009/02/16 13:21:15 $
 *  $Revision: 1.2 $
 *  \author Paolo Ronchese INFN Padova
 *
 */

//-----------------------
// This Class' Header --
//-----------------------
#include "CondTools/DT/interface/DTTPGParametersHandler.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "CondFormats/DTObjects/interface/DTTPGParameters.h"

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
DTTPGParametersHandler::DTTPGParametersHandler( const edm::ParameterSet& ps ):
 dataTag(   ps.getParameter<std::string>  (  "tag" ) ),
 fileName(  ps.getParameter<std::string>  ( "file" ) ),
 runNumber( ps.getParameter<unsigned int> (  "run" ) ) {
}

//--------------
// Destructor --
//--------------
DTTPGParametersHandler::~DTTPGParametersHandler() {
}

//--------------
// Operations --
//--------------
void DTTPGParametersHandler::getNewObjects() {

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

  DTTPGParameters* tSync = new DTTPGParameters( dataTag );

  int status = 0;
  std::ifstream ifile( fileName.c_str() );
//  int cx = 32;
//  ifile >> cx;
//  tSync->setClock( cx );
  int whe;
  int sta;
  int sec;
  int   cl;
  float ph;
  while ( ifile >> whe
                >> sta
                >> sec
                >> cl
                >> ph ) {
    status = tSync->set( whe, sta, sec, cl, ph,
                         DTTimeUnits::ns );
    std::cout << whe << " "
              << sta << " "
              << sec << " "
              << cl << " "
              << ph << "  -> ";                
    std::cout << "insert status: " << status << std::endl;
  }

/*
  unsigned int runf = irun;
  unsigned int runl = 0xffffffff;
  popcon::IOVPair iop = { runf, runl };
  std::cout << "APPEND NEW OBJECT: "
            << runf << " " << runl << " " << tSync << std::endl;
  m_to_transfer->push_back( std::make_pair( tSync, iop ) );
*/

  //for each payload provide IOV information (say in this case we use since)
  cond::Time_t snc = runNumber;
  if ( runNumber > last )
       m_to_transfer.push_back( std::make_pair( tSync, snc ) );
  else
       std::cout << "More recent data already present - skipped" << std::endl;

  return;

}


std::string DTTPGParametersHandler::id() const {
  return dataTag;
}


