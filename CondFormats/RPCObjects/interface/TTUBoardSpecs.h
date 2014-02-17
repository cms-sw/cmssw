// $Id: TTUBoardSpecs.h,v 1.1 2009/01/28 12:54:41 aosorio Exp $
#ifndef CONFIGCODE_TTUBOARDSPECS_H 
#define CONFIGCODE_TTUBOARDSPECS_H 1

// Include files
#include "CondFormats/RPCObjects/interface/RPCTechTriggerConfig.h"
#include <vector>
#include <string>

/** @class TTUBoardSpecs TTUBoardSpecs.h ConfigCode/TTUBoardSpecs.h
 *  
 *
 *  @author Andres Osorio
 *  @date   2008-12-15
 */
class TTUBoardSpecs {
public: 
  /// Standard constructor
  TTUBoardSpecs( ); 
  
  virtual ~TTUBoardSpecs( ); ///< Destructor
  
  class TTUBoardConfig : public RPCTechTriggerConfig {
  public: 
    /// Standard constructor
    TTUBoardConfig( ) : RPCTechTriggerConfig() {}; 
    
    int m_Firmware;
    int m_LengthOfFiber;
    int m_Delay;
    int m_MaxNumWheels;
    int m_Wheel1Id;
    int m_Wheel2Id;
    int m_TrackLength;
        
    std::vector<int> m_MaskedSectors;
    std::vector<int> m_ForcedSectors;
     
    std::string m_LogicType;

  };
  
  std::vector<TTUBoardConfig> m_boardspecs;
  
};
#endif // CONFIGCODE_TTUBOARDSPECS_H
