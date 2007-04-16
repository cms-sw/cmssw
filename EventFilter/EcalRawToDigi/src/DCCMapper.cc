/* 
 * \class DCCMapper
 * \file DCCMapper.cc
 *
 * Preliminary mapper of the DCC boards to the supermodule id
 * Reads data from an ascii file and enables the direct access to the DCC id
 * from the SM id and vice-versa. Retrieves the full map on request.
 *
 * $Date: 2006/06/19 16:08:44 $
 * $Revision: 1.2 $
 * \author P.Silva (psilva@cern.
 */

#include "EventFilter/EcalRawToDigi/src/DCCMapper.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

using namespace std;

/*----------------------------------------------------*/
/* DCCMapper::DCCMapper()                             */
/* class default constructor                          */
/*----------------------------------------------------*/

DCCMapper::DCCMapper()
{
  pathToMapFile_=string("");
}

/*----------------------------------------------------*/
/* DCCMapper::setDCCMapFilePath(string aPath_)        */
/* tries to set file path to a given string           */
/* returns true on sucess                             */
/*----------------------------------------------------*/

bool DCCMapper::setDCCMapFilePath(string aPath_)
{

  //try to open a dccMapFile in the given path
  ifstream dccMapFile_(aPath_.c_str());

  //if not successful return false
  if(!dccMapFile_.is_open()) return false;

  //else close file and accept given path
  dccMapFile_.close();
  pathToMapFile_ = aPath_;

  return true;
}

/*----------------------------------------------------*/
/* DCCMapper::readDCCMapFile()                        */
/* tries to read a DCC map file                       */
/* returns true on sucess                             */
/*----------------------------------------------------*/

bool DCCMapper::readDCCMapFile()
{
  //try to open a dccMapFile in the given path
  ifstream dccMapFile_(pathToMapFile_.c_str());
  
  //if not successful return false
  if(!dccMapFile_.is_open()) return false;
  
  char lineBuf_[100];
  ulong SMId_,DCCId_;
  // loop while extraction from file is possible
  while (dccMapFile_.good()) {
    dccMapFile_.getline(lineBuf_,10);       //read line from file
    sscanf(lineBuf_,"%lu:%lu",&SMId_,&DCCId_);
    myDCCMap_[SMId_] = DCCId_;
  }
  
  return true;
  
}

/*----------------------------------------------------*/
/* DCCMapper::readDCCMapFile(string aPath_)           */
/* tries to read a DCC map file from a given path     */
/* location and returns true on success               */
/*----------------------------------------------------*/

bool DCCMapper::readDCCMapFile(string aPath_)
{
  //test if path is good
  edm::FileInPath eff(aPath_);
  
  if(!setDCCMapFilePath(eff.fullPath())) return false;

  //read DCC map file
  readDCCMapFile();
  return true;
}

/*----------------------------------------------------*/
/* DCCMapper::getDCCId(ulong aSMId_)                  */
/* returns a DCC id corresponding to a SM id          */
/* or 0 if not found                                  */
/*----------------------------------------------------*/
ulong DCCMapper::getDCCId(ulong aSMId_) const
{
  //get iterator for SM id
  map<ulong ,ulong>::const_iterator it = myDCCMap_.find(aSMId_);

  //check if SMid exists and return DCC id
  if(it!= myDCCMap_.end()) return it->second;
 
  //error return
  edm::LogError("EcalDCCMapper") << "DCC requested for SM id: " << aSMId_ << " not found" ;
  return 0;
}

/*----------------------------------------------------*/
/* DCCMapper::getSMId(ulong aDCCId_)                  */
/* returns a SM id corresponding to a DCC id          */
/* or 0 if not found                                  */
/*----------------------------------------------------*/

ulong DCCMapper::getSMId(ulong aDCCId_) const 
{
  //get iterator map
  map<ulong ,ulong>::const_iterator it;

  //try to find SM id for given DCC id
  for(it = myDCCMap_.begin(); it != myDCCMap_.end(); it++)
    if(it->second == aDCCId_) 
      return it->first;

  //error return
  edm::LogError("EcalDCCMapper") << "SM requested DCC id: " << aDCCId_ << " not found" << endl;
  return 0;
}

/*----------------------------------------------------*/
/* DCCMapper::~DCCMapper()                            */
/* class destructor                                   */
/*----------------------------------------------------*/
DCCMapper::~DCCMapper()
{
  pathToMapFile_.clear();
}


/*--------------------------------------------------------------------------*/
/* std::ostream &operator<< (std::ostream& o, const DCCMapper &aDCCMapper_) */
/* overrides default << operator to print DCCMapper class information       */
/*--------------------------------------------------------------------------*/
std::ostream &operator<< (std::ostream& o, const DCCMapper &aDCCMapper_) 
{
  //print class information
  o << "---------------------------------------------------------" << endl;

  if(aDCCMapper_.pathToMapFile_.size() < 1){
    o << "No correct input for DCC map has been given yet..." << endl;
  }
  else{
    o << "DCC Map (Map file: " << aDCCMapper_.pathToMapFile_ << " )" << endl
      << "SM id\t\tDCCid " << endl;

    //get DCC map and iterator
    map<ulong ,ulong > aMap;
    aMap=aDCCMapper_.myDCCMap_;
    map<ulong ,ulong >::iterator iter;

    //print info contained in map
    for(iter = aMap.begin(); iter != aMap.end(); iter++)
      o << iter->first << "\t\t" << iter->second << endl;
  }

  o << "---------------------------------------------------------" << endl;
  return o;
}

