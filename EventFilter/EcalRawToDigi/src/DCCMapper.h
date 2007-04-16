/* 
 * \class DCCMapper
 * \file DCCMapper.hh
 *
 * Preliminary mapper of the DCC boards to the supermodule id
 * Reads data from an ascii file and enables the direct access to the DCC id
 * from the SM id and vice-versa. Retrieves the full map on request.
 *
 * $Date: 2006/06/19 16:08:44 $
 * $Revision: 1.2 $
 * \author P.Silva (psilva@cern.ch)
 */

#ifndef _DCCMAPPER_HH_
#define _DCCMAPPER_HH_ 1

#include <iostream>                     //C++
#include <fstream>
//#include <ifstream>
#include <string>
#include <map>


class DCCMapper{
  
public:
  /**
   * Default constructor
   */
  DCCMapper();

  /**
   * Destructor
   */
  ~DCCMapper();

  /**
   * Receives a string with a path and checks if file is accessible
   */
  bool setDCCMapFilePath(std::string );

  /**
   * Retrieves current path do the map file
   */
  std::string getDCCMapFilePath() const { return pathToMapFile_; };
  
  /**
   * Read map file (returns false if an error ocurred)
   */
  bool readDCCMapFile();
  bool readDCCMapFile(std::string );

  /**
   * Get methods for DCCId/SMId and map
   */

  ulong getDCCId(ulong ) const ;
  ulong getSMId(ulong ) const ;
  const std::map<ulong ,ulong>& getDCCMap() const { return myDCCMap_; };

  /**
   * Print current map
   */

  friend std::ostream& operator<< (std::ostream &o, const DCCMapper& aDCCMapper);



private:

  std::string pathToMapFile_;

  std::map<ulong ,ulong> myDCCMap_;

};

#endif
