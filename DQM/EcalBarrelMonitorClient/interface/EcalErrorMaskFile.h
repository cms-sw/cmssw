// $Id: EcalErrorMaskFile.h,v 1.9 2007/01/18 07:55:23 dellaric Exp $

/*!
  \file EcalErrorMaskFile.h
  \brief Error mask from text file
  \author B. Gobbo 
  \version $Revision: 1.9 $
  \date $Date: 2007/01/18 07:55:23 $
*/

#ifndef EcalErrorMaskFile_H
#define EcalErrorMaskFile_H

#include <string>
#include <map>
#include "OnlineDB/EcalCondDB/interface/RunCrystalErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunTTErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunPNErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunMemChErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/RunMemTTErrorsDat.h"
#include "OnlineDB/EcalCondDB/interface/EcalLogicID.h"

class EcalErrorMaskFile {

 public:

  static void readFile( std::string inFile ) throw( std::runtime_error );

  static void fetchDataSet( std::map< EcalLogicID, RunCrystalErrorsDat>* fillMap ) throw( std::runtime_error );
  static void fetchDataSet( std::map< EcalLogicID, RunTTErrorsDat>* fillMap ) throw( std::runtime_error );
  static void fetchDataSet( std::map< EcalLogicID, RunPNErrorsDat>* fillMap ) throw( std::runtime_error );
  static void fetchDataSet( std::map< EcalLogicID, RunMemChErrorsDat>* fillMap ) throw( std::runtime_error );
  static void fetchDataSet( std::map< EcalLogicID, RunMemTTErrorsDat>* fillMap ) throw( std::runtime_error );

 private:

  static bool done_;
  static std::string inFile_;

  static std::map<EcalLogicID, RunCrystalErrorsDat> mapCrystalErrors_;
  static std::map<EcalLogicID, RunTTErrorsDat>      mapTTErrors_;
  static std::map<EcalLogicID, RunPNErrorsDat>      mapPNErrors_;
  static std::map<EcalLogicID, RunMemChErrorsDat>   mapMemChErrors_;
  static std::map<EcalLogicID, RunMemTTErrorsDat>   mapMemTTErrors_;

  static void clearComments_( char* line );
  static void clearFinalBlanks_( char* line );

};

#endif // EcalErrorMaskFile_H
