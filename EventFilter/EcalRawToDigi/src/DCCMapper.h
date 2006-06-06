/* 
 * \class DCCMapper
 * \file DCCMapper.hh
 *
 * Preliminary mapper of the DCC boards to the supermodule id
 * Reads data from an ascii file and enables the direct access to the DCC id
 * from the SM id and vice-versa. Retrieves the full map on request.
 *
 * $Date: 2006/05/17 9:34:00$
 * $Revision: 1.0 $
 * \author P.Silva (psilva@cern.ch)
 */

#ifndef _DCCMAPPER_HH_
#define _DCCMAPPER_HH_ 1

#include <iostream>                     //C++
#include <fstream>
//#include <ifstream>
#include <string>
#include <map>

using namespace std;

class DCCMapper{
  
public:
  /**
   * Default constructor
   */
  DCCMapper();

  /**
   * Receives a string with a path and checks if file is accessible
   */
  bool setDCCMapFilePath(string );

  /**
   * Retrieves current path do the map file
   */
  string getDCCMapFilePath();
  
  /**
   * Read map file (returns false if an error ocurred)
   */
  bool readDCCMapFile();
  bool readDCCMapFile(string );

  /**
   * Get methods for DCCId/SMId and map
   */
  ulong getDCCId(ulong );
  ulong getSMId(ulong );
  map<ulong ,ulong> getDCCMap();

  /**
   * Print current map
   */
  friend std::ostream& operator<< (std::ostream &o, const DCCMapper& aDCCMapper);

  /**
   * Destructor
   */
  ~DCCMapper();

private:

  string pathToMapFile_;
  map<ulong ,ulong > myDCCMap_;

};

//inline methods to return private variables
inline string DCCMapper::getDCCMapFilePath() { return pathToMapFile_; }
inline map<ulong ,ulong > DCCMapper::getDCCMap() {return myDCCMap_;}

#endif
