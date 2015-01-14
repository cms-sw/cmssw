#ifndef PixelDELAY25CALIB_h
#define PixelDELAY25CALIB_h
/**
*   \file CalibFormats/SiPixelObjects/interface/PixelDelay25Calib.h
*   \brief This class manages data and files used in the Delay25 calibration
*
*   A longer explanation will be placed here later
*/
#include <vector>
#include <string>
#include <set>
#include <fstream>
#include "CalibFormats/SiPixelObjects/interface/PixelCalibBase.h"
#include "CalibFormats/SiPixelObjects/interface/PixelConfigBase.h"

namespace pos{
/*!  \ingroup ConfigurationObjects "Configuration Objects"
*    
*  @{
*
*  \class PixelDelay25Calib PixelDelay25Calib.h
*  \brief This class manages data and files used in the Delay25 calibration
*/
  class PixelDelay25Calib : public PixelCalibBase, public PixelConfigBase {

  public:
  
    PixelDelay25Calib(std::string);
    PixelDelay25Calib(std::vector<std::vector<std::string> > &);
    ~PixelDelay25Calib();

    virtual void writeASCII(std::string dir="") const;
    void 	 writeXML(        pos::PixelConfigKey key, int version, std::string path) const {;}
    virtual void writeXMLHeader(  pos::PixelConfigKey key, 
				  int version, 
				  std::string path, 
				  std::ofstream *out,
				  std::ofstream *out1 = NULL,
				  std::ofstream *out2 = NULL
				  ) const ;
    virtual void writeXML( 	  std::ofstream *out,			     	   			    
			   	  std::ofstream *out1 = NULL ,
			   	  std::ofstream *out2 = NULL ) const ;
    virtual void writeXMLTrailer( std::ofstream *out, 
				  std::ofstream *out1 = NULL,
				  std::ofstream *out2 = NULL
				  ) const ;

    std::set<std::string>& portcardList() {return portcardNames_;}
    bool allPortcards() {return allPortcards_;}
    bool allModules() {return allModules_;}
    int getGridSize() {return gridSize_;}
    int getGridSteps() {return gridSteps_;}
    int getNumberTests() {return numTests_;}
    int getRange() {return range_;}
    int getOrigSDa() {return origSDa_;}
    int getOrigRDa() {return origRDa_;}
    int getCommands() {return commands_;}
    void openFiles(std::string portcardName, std::string moduleName, 
		   std::string path="");
    void writeSettings(std::string portcardName, std::string moduleName);
    void writeFiles(std::string tmp);
    void writeFiles(int currentSDa, int currentRDa, int number);
    void closeFiles();

    // Added by Dario April 28th, 2010
    std::string getStreamedContent(void) const {return calibFileContent_;} ;

  private:

    std::set<std::string> portcardNames_;
    bool allPortcards_, allModules_;
    int origSDa_, origRDa_, range_, gridSize_, gridSteps_, numTests_, commands_;
    std::ofstream graphout_;
    std::string graph_;

    // Added by Dario April 28th, 2010
    std::string calibFileContent_ ;
  };
}
/* @} */
#endif
