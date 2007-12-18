#ifndef ECALTPGCONDDBAPP_H
#define ECALTPGCONDDBAPP_H

#include <iostream>
#include <string>
#include <sstream>

#include "OnlineDB/EcalCondDB/interface/EcalCondDBInterface.h"
#include "OnlineDB/EcalCondDB/interface/RunDat.h"
#include "OnlineDB/EcalCondDB/interface/RunList.h"


class EcalTPGCondDBApp {
public:

  /**
   *   App constructor; Makes the database connection
   */
  EcalTPGCondDBApp(string host, string sid, string user, string pass, int port) ;
  EcalTPGCondDBApp(string sid, string user, string pass) ;

  /**
   *  App destructor;  Cleans up database connection
   */
  ~EcalTPGCondDBApp() ;

  inline std::string to_string( char value[])
  {
    std::ostringstream streamOut;
    streamOut << value;
    return streamOut.str();    
  }
  

  int writePedestals(int runa, int runb ) ;
  void readPedestals(int iconf_req ) ;
  void writeLUT() ;
  void writeWeights() ;


private:
  EcalTPGCondDBApp();  // hidden default constructor
  EcalCondDBInterface* econn;
  
  uint64_t startmicros;
  uint64_t endmicros;
  run_t startrun;
  run_t endrun;

  void printTag( const RunTag* tag) const ;
  void printIOV( const RunIOV* iov) const ;

};

#endif

