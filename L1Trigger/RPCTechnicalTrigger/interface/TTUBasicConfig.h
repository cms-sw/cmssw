// $Id: TTUBasicConfig.h,v 1.2 2009/07/01 22:52:06 aosorio Exp $
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
  /// Standard constructor
  TTUBasicConfig( ) { };

  TTUBasicConfig( const char * );
  
  TTUBasicConfig( const TTUBoardSpecs * );
  
  virtual ~TTUBasicConfig( ); ///< Destructor

  bool initialise( int , int );

  void preprocess( TTUInput & );
  
protected:
  
private:

  std::vector<int> m_vecmask;
  std::vector<int> m_vecforce;

  bool m_debug;
    
};
#endif // INTERFACE_TTUBASICCONFIG_H
