#include "DCCMapper.hh"

int main(){
  DCCMapper myMap;

  myMap.readDCCMapFile("DCCMap.txt");
  
  myMap.getDCCId(1);
  myMap.getDCCId(2);
  myMap.getDCCId(3);
  myMap.getDCCId(4);
  myMap.getDCCId(6);

  myMap.getSMId(1);
  myMap.getSMId(2);
  myMap.getSMId(3);
  myMap.getSMId(4);
  myMap.getSMId(6);

  cout << myMap;

  return 0;
}

/* 
 * \class DCCMapper
 * \file DCCMapper.cc
 *
 * Preliminary mapper of the DCC boards to the supermodule id
 * Reads data from an ascii file and enables the direct access to the DCC id
 * from the SM id and vice-versa. Retrieves the full map on request.
 *
 * $Date: 2006/05/17 9:34:00$
 * $Revision: 1.0 $
 * \author P.Silva (psilva@cern.
 */



/*----------------------------------------------------*/
/* DCCMapper::DCCMapper()                             */
/* class default constructor                          */
/*----------------------------------------------------*/
DCCMapper::DCCMapper(){
  pathToMapFile_=string("");
}

/*----------------------------------------------------*/
/* DCCMapper::setDCCMapFilePath(string aPath_)        */
/* tries to set file path to a given string           */
/* returns true on sucess                             */
/*----------------------------------------------------*/
bool DCCMapper::setDCCMapFilePath(string aPath_){

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
bool DCCMapper::readDCCMapFile(){
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
bool DCCMapper::readDCCMapFile(string aPath_){
  //test if path is good
  if(!setDCCMapFilePath(aPath_)) return false;

  //read DCC map file
  readDCCMapFile();
  return true;
}

/*----------------------------------------------------*/
/* DCCMapper::getDCCId(ulong aSMId_)                  */
/* returns a DCC id corresponding to a SM id          */
/* or 0 if not found                                  */
/*----------------------------------------------------*/
ulong DCCMapper::getDCCId(ulong aSMId_){
  //get iterator for SM id
  map<ulong ,ulong>::iterator iter = myDCCMap_.find(aSMId_);

  //check if SMid exists and return DCC id
  if(iter!= myDCCMap_.end()) return iter->second;
 
  //error return
  cout << "DCC requested for SM id: " << aSMId_ << " not found" << endl;
  return 0;
}

/*----------------------------------------------------*/
/* DCCMapper::getSMId(ulong aDCCId_)                  */
/* returns a SM id corresponding to a DCC id          */
/* or 0 if not found                                  */
/*----------------------------------------------------*/
ulong DCCMapper::getSMId(ulong aDCCId_){
  //get iterator map
  map<ulong ,ulong>::iterator iter;

  //try to find SM id for given DCC id
  for(iter = myDCCMap_.begin(); iter != myDCCMap_.end(); iter++)
    if(iter->second == aDCCId_) 
      return iter->first;

  //error return
  cout << "SM requested DCC id: " << aDCCId_ << " not found" << endl;
  return 0;
}

/*----------------------------------------------------*/
/* DCCMapper::~DCCMapper()                            */
/* class destructor                                   */
/*----------------------------------------------------*/
DCCMapper::~DCCMapper(){
  pathToMapFile_.clear();
}


/*--------------------------------------------------------------------------*/
/* std::ostream &operator<< (std::ostream& o, const DCCMapper &aDCCMapper_) */
/* overrides default << operator to print DCCMapper class information       */
/*--------------------------------------------------------------------------*/
std::ostream &operator<< (std::ostream& o, const DCCMapper &aDCCMapper_) {
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
      cout << iter->first << "\t\t" << iter->second << endl;
  }

  o << "---------------------------------------------------------" << endl;
  return o;
}

