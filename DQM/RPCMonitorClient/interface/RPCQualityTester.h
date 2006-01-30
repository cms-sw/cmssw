#ifndef RPCQualityTester_H
#define RPCQualityTester_H

/** \class RPCQualityTester
 * *
 *  Class that handles the RPC Quality Tests
 * 
 *  $Date: 2006/01/25 16:28:17 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
  */

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include <vector>
#include <fstream>
#include <string>

class RPCQualityTester
{
 public:
  
  ///creator
  RPCQualityTester();
  
  ///destructor
  ~RPCQualityTester();

  /// Set up Quality Tests
  void SetupTests(MonitorUserInterface * mui) ;

  /// Attaches Quality Tests to ME's
  void AttachTests(MonitorUserInterface * mui) ;
  
  /// Set up Quality Tests from Db and Sterts tests configuration
  void SetupTestsFromDB(MonitorUserInterface * mui) ;
 
  /// Configures Tets
  void SetContentsXRangeROOTTest(MonitorUserInterface * mui,char[20] , float , float[5] ) ;
  
  
  /// Check Status of Quality Tests
  void CheckTests(MonitorUserInterface * mui);
  

 private:
  
  
  bool printout;
  std::string qtest1;
  std::vector<std::string> qTests;
  std::ofstream logFile;
  
};

#endif
