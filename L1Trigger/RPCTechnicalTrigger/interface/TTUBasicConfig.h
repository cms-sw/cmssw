#ifndef INTERFACE_TTUBASICCONFIG_H 
#define INTERFACE_TTUBASICCONFIG_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/TTUConfiguration.h"

#include "CondFormats/RPCObjects/interface/TTUBoardSpecs.h"

/** @class TTUBasicConfig TTUBasicConfig.h interface/RPCBasicConfig.h
 *  
 *
 *  @author Andres Osorio
 *  @date   2008-10-29
 */
class TTUBasicConfig : public TTUConfiguration {
public:
  TTUBasicConfig( const char * );
  
  TTUBasicConfig( const TTUBoardSpecs * );
  
  ~TTUBasicConfig( ) override; ///< Destructor

  bool initialise( int , int ) override;

  void preprocess( TTUInput & ) override;
  
protected:
  
private:

  std::vector<int> m_vecmask;
  std::vector<int> m_vecforce;

  bool m_debug;
    
};
#endif // INTERFACE_TTUBASICCONFIG_H
