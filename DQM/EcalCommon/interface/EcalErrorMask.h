// $Id: EcalErrorMask.h,v 1.13 2008/06/23 16:38:29 dellaric Exp $

/*!
  \file EcalErrorMask.h
  \brief Error mask from text file or database
  \author B. Gobbo 
  \version $Revision: 1.13 $
  \date $Date: 2008/06/23 16:38:29 $
*/

#ifndef EcalErrorMask_H
#define EcalErrorMask_H

#include <cstdlib>
#include <string>
#include <map>

class EcalCondDBInterface;

class EcalLogicID;

class RunCrystalErrorsDat;
class RunTTErrorsDat;
class RunPNErrorsDat;
class RunMemChErrorsDat;
class RunMemTTErrorsDat;
class RunIOV;

class EcalErrorMask {

 public:

  static void readFile( std::string& inFile, bool debug = false, bool verifySyntax = false ) throw( std::runtime_error );
  static void writeFile( std::string& outFile ) throw( std::runtime_error );

  static void readDB( EcalCondDBInterface* eConn, RunIOV* runIOV ) throw( std::runtime_error );
  static void writeDB( EcalCondDBInterface* eConn, RunIOV* runIOV );

  static std::map<EcalLogicID, RunCrystalErrorsDat> mapCrystalErrors_;
  static std::map<EcalLogicID, RunTTErrorsDat>      mapTTErrors_;
  static std::map<EcalLogicID, RunPNErrorsDat>      mapPNErrors_;
  static std::map<EcalLogicID, RunMemChErrorsDat>   mapMemChErrors_;
  static std::map<EcalLogicID, RunMemTTErrorsDat>   mapMemTTErrors_;

 private:

  static int runNb_;

  static void clearComments_( char* line );
  static void clearFinalBlanks_( char* line );

};

#endif // EcalErrorMask_H
