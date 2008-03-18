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
  class PixelDelay25Calib : public PixelCalibBase, public PixelConfigBase{

  public:
  
    PixelDelay25Calib(std::string);
    ~PixelDelay25Calib();

    virtual void writeASCII(std::string dir="") const;


    virtual std::string mode() {return mode_;}
    std::set<std::string>& portcardList() {return portcardNames_;}
    bool allModules() {return allModules_;}
    int getGridSize() {return gridSize_;}
    int getGridSteps() {return gridSteps_;}
    int getNumberTests() {return numTests_;}
    int getNextOrigSDa(int n);
    int getNextOrigRDa(int n);
    void openFiles(std::string portcardName, std::string moduleName);
    void writeSettings(std::string portcardName, std::string moduleName);
    void writeFiles(std::string tmp);
    void writeFiles(int currentSDa, int currentRDa, int number);
    void closeFiles();
    void getCandidatePoints();
    int getNumCandidatePoints() {return numCandidatePoints_;}
    int getNextCandidateSDa(int n);
    int getNextCandidateRDa(int n);
    int getStableRange() {return stableRange_;}
    void makeNeighbors(int SDa, int RDa);
    int getNumNeighbors() {return numNeighbors_;}
    int getNextNeighborSDa(int n);
    int getNextNeighborRDa(int n);

  private:
    std::string mode_;
    std::set<std::string> portcardNames_;
    bool allModules_;
    int origSDa_, origRDa_, range_, gridSize_;
    int numTests_, stableRange_, stableShape_;
    int gridSteps_, numShifts_, numCandidatePoints_, numNeighbors_;
    std::vector<int> vecOrigSDa_, vecOrigRDa_, vecShifts_;
    std::vector<int> vecCandidateSDa_, vecCandidateRDa_;
    std::vector<int> vecNeighborSDa_, vecNeighborRDa_;
    std::ofstream graphout_;
    std::ofstream goodout_;
    std::string graph_, good_;

  };
}
/* @} */
#endif
