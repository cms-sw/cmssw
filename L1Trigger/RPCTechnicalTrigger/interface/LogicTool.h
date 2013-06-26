// $Id: LogicTool.h,v 1.3 2009/08/09 11:11:36 aosorio Exp $
#ifndef LOGICTOOL_H 
#define LOGICTOOL_H 1

// Include files
#include "L1Trigger/RPCTechnicalTrigger/interface/LogicFactory.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/LogicImp.h"

#include <ostream>
#include <vector>

/** @class LogicTool LogicTool.h
 *  
 *
 *  @author Andres Osorio
 *
 *  email: aosorio@uniandes.edu.co
 *
 *  @date   2008-10-12
 */
template <class GenLogic> 
class LogicTool {
public: 
  /// Standard constructor
  LogicTool( ) { }; 
  
  virtual ~LogicTool( ) { 
    m_logkeys.clear();
  }; ///< Destructor

  ///...

  typedef LogicFactory< GenLogic, std::string> RBCLogicType;
  
  GenLogic * (*createlogic) ();
    
  RBCLogicType m_rbclogic;
  
  bool initialise() 
  {
    bool status(true);

    //...
    std::string key = std::string("ChamberORLogic");
    createlogic = (GenLogic * (*)()) &createChamberORLogic;
    status = m_rbclogic.Register( key , createlogic );
    
    m_logkeys.push_back( key );
        
    key = std::string("TestLogic");
    createlogic = (GenLogic * (*)()) &createTestLogic;
    status = m_rbclogic.Register( key , createlogic );
    
    m_logkeys.push_back( key );

    key = std::string("PatternLogic");
    createlogic = (GenLogic * (*)()) &createPatternLogic;
    status = m_rbclogic.Register( key , createlogic );
    //...
    
    m_logkeys.push_back( key );

    key = std::string("TrackingAlg");
    createlogic = (GenLogic * (*)()) &createTrackingAlg;
    status = m_rbclogic.Register( key , createlogic );

    m_logkeys.push_back( key );
    
    key = std::string("SectorORLogic");
    createlogic = (GenLogic * (*)()) &createSectorORLogic;
    status = m_rbclogic.Register( key , createlogic );
    
    m_logkeys.push_back( key );

    key = std::string("TwoORLogic");
    createlogic = (GenLogic * (*)()) &createTwoORLogic;
    status = m_rbclogic.Register( key , createlogic );

    m_logkeys.push_back( key );

    key = std::string("WedgeORLogic");
    createlogic = (GenLogic * (*)()) &createWedgeORLogic;
    status = m_rbclogic.Register( key , createlogic );
    
    m_logkeys.push_back( key );

    key = std::string("PointingLogic");
    createlogic = (GenLogic * (*)()) &createPointingLogic;
    status = m_rbclogic.Register( key , createlogic );
    
    m_logkeys.push_back( key );
    
    return status;
    
  };
  
  
  GenLogic * retrieve( const std::string & _logic_name ) 
  {
    GenLogic * _obj;
    _obj = m_rbclogic.CreateObject( _logic_name );
    return _obj;
  };
  
  bool endjob()
  {
    bool status(true);
    typename std::vector<std::string>::iterator itr = m_logkeys.begin();
    while ( itr != m_logkeys.end() ) 
    {
      status = status && ( m_rbclogic.Unregister( (*itr) ) );
      ++itr;
    }
    return status;
    
  };
  
  
protected:
  
private:
  
  typename std::vector<std::string> m_logkeys;
  
};
#endif // LOGICTOOL_H
