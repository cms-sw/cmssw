// $Id: EcalErrorMask.h,v 1.11 2008/03/14 14:38:56 dellaric Exp $

/*!
  \file EcalErrorMask.h
  \brief Error mask from text file or database
  \author B. Gobbo 
  \version $Revision: 1.11 $
  \date $Date: 2008/03/14 14:38:56 $
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

  static void fetchDataSet( std::map< EcalLogicID, RunCrystalErrorsDat>* fillMap );
  static void fetchDataSet( std::map< EcalLogicID, RunTTErrorsDat>* fillMap );
  static void fetchDataSet( std::map< EcalLogicID, RunPNErrorsDat>* fillMap );
  static void fetchDataSet( std::map< EcalLogicID, RunMemChErrorsDat>* fillMap );
  static void fetchDataSet( std::map< EcalLogicID, RunMemTTErrorsDat>* fillMap );

 private:

  static bool done_;

  static int runNb_;

  static std::map<EcalLogicID, RunCrystalErrorsDat> mapCrystalErrors_;
  static std::map<EcalLogicID, RunTTErrorsDat>      mapTTErrors_;
  static std::map<EcalLogicID, RunPNErrorsDat>      mapPNErrors_;
  static std::map<EcalLogicID, RunMemChErrorsDat>   mapMemChErrors_;
  static std::map<EcalLogicID, RunMemTTErrorsDat>   mapMemTTErrors_;

  static void clearComments_( char* line );
  static void clearFinalBlanks_( char* line );

};

#endif // EcalErrorMask_H
