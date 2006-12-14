// $Id: $

/*!
  \file EcalErrorMaskFile.h
  \brief Error mask from text file
  \author B. Gobbo 
  \version $Revision: $
  \date $Date: $
*/

#ifndef EcalErrorMaskFile_H
#define EcalErrorMaskFile_H

#include <string>
#include <map>
#include "OnlineDB/EcalCondDB/interface/MonCrystalStatusDat.h"
#include "OnlineDB/EcalCondDB/interface/MonPNStatusDat.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class EcalErrorMaskFile {

 public:

  static void readFile( std::string inFile ) throw( std::runtime_error );

  template <class T> static void fetchData( std::map< EcalLogicID, T>& fillMap ) throw( std::runtime_error );

 private:

  static bool done_;
  static std::string inFile_;
  static std::map<EcalLogicID, MonCrystalStatusDat> mapMCSD_;
  static std::map<EcalLogicID, MonPNStatusDat>      mapMPSD_;

  static void clearComments_( char* line );
  static void clearFinalBlanks_( char* line );

};

#endif // EcalErrorMaskFile_H
