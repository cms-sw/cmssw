// $Id: RBCBoardSpecs.h,v 1.1 2009/01/28 12:54:41 aosorio Exp $
#ifndef CONFIGCODE_RBCBOARDSPECS_H 
#define CONFIGCODE_RBCBOARDSPECS_H 1

// Include files
#include <vector>
#include <string>

/** @class RBCBoardSpecs RBCBoardSpecs.h ConfigCode/RBCBoardSpecs.h
 *  
 *
 *  @author Andres Osorio
 *  @date   2008-12-15
 */
class RBCBoardSpecs {
public: 

  RBCBoardSpecs( ); 
  virtual ~RBCBoardSpecs( ); ///< Destructor
 
  class RBCBoardConfig {
  public:
    
    RBCBoardConfig( ) {};
    virtual ~RBCBoardConfig( ) {}; ///< Destructor
    
    int m_Firmware;
    int m_WheelId;
    int m_Latency;
    int m_MayorityLevel;
    
    std::vector<int> m_MaskedOrInput;
    std::vector<int> m_ForcedOrInput;
    
    std::string m_LogicType;

  };
  
  std::vector<RBCBoardConfig> v_boardspecs;
  
};
#endif // CONFIGCODE_RBCBOARDSPECS_H
