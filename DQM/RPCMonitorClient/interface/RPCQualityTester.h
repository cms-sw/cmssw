#ifndef RPCQualityTester_H
#define RPCQualityTester_H

/** \class RPCQualityTester
 * *
 *  Class that handles the RPC Quality Tests
 * 
 *  $Date: 2006/01/30 17:38:41 $
 *  $Revision: 1.1 $
 *  \author Ilaria Segoni
  */

#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include <vector>
#include <fstream>
#include <string>
#include <map>

class RPCQualityTester
{
 public:
  
  ///creator
  RPCQualityTester();
  
  ///destructor
  ~RPCQualityTester();

  /// Set up Quality Tests
  void SetupTests(MonitorUserInterface * mui) ;

  /// Set up Quality Tests from Db and Sterts tests configuration
  void SetupTestsFromTextFile(MonitorUserInterface * mui) ;
 
  /// Fills map<QTestName,MonitorElement>
  void LinkTeststoME() ;
 
  /// Attaches Quality Tests to ME's
  void AttachRunTests(MonitorUserInterface * mui) ;
  
  /// Configures Test of type ContentsXRangeROOT 
  void SetContentsXRangeROOTTest(MonitorUserInterface * mui,char[20] , float , float[5] ) ;

  /// Check Status of Quality Tests
  void CheckTests(MonitorUserInterface * mui);
  

 private:
  
  
  bool printout;
  std::vector<std::string> qTests;
  std::map< std::string , std::vector<std::string> > qTestToMEMap;
  std::ofstream logFile;
  
};

#endif
