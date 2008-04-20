#ifndef SiPixelQTestsParser_H
#define SiPixelQTestsParser_H

/** \class SiPixelQTestsParser
 * *
 *  Class that handles the SiPixel Quality Tests
 * 
 *  $Date: 2007/04/10 20:56:26 $
 *  $Revision: 1.1 $
 *  \author Mia Tosi
  */

#include "DQMServices/ClientConfig/interface/DQMParserBase.h"
#include <vector>
#include <fstream>
#include <string>
#include <map>

class QTestParameterNames;

class SiPixelQTestsParser : public DQMParserBase {

 public:
  

  // Constructor
  SiPixelQTestsParser();
  
  // Destructor
  ~SiPixelQTestsParser();

  /// Returns the Quality Tests list with their parameters obtained from the xml file
  bool getAllQTests(std::map<std::string, std::map<std::string, std::string> >& mapQTests);
  /// Returns the map between the MonitoElemnt and the list of tests requested for it
  bool monitorElementTestsMap(std::map<std::string, std::vector<std::string> >& mapMEsQTests);

 private:
  std::map<std::string, std::string> getParams(xercesc::DOMElement* qtestElement, 
					       std::string test);
  
  std::string testActivationOFF_;
  std::string testON_;
  
  QTestParameterNames * qtestParamNames;
  
};

#endif
