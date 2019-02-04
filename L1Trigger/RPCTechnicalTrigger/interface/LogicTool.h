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
  LogicTool( ) {
  }
  
  ///...

  std::unique_ptr<GenLogic> retrieve( const std::string & _logic_name ) 
  {
    return  rbclogic().CreateObject( _logic_name );
  };
  
protected:
  
private:
  using RBCLogicType =  LogicFactory< GenLogic, std::string>;
  
  static RBCLogicType initialise() 
  {
    GenLogic * (*createlogic) ();
    bool status(true);

    RBCLogicType rbclogic;
    //...
    std::string key = std::string("ChamberORLogic");
    createlogic = (GenLogic * (*)()) &createChamberORLogic;
    status = rbclogic.Register( key , createlogic );
    
    key = std::string("TestLogic");
    createlogic = (GenLogic * (*)()) &createTestLogic;
    status = rbclogic.Register( key , createlogic );
    
    key = std::string("PatternLogic");
    createlogic = (GenLogic * (*)()) &createPatternLogic;
    status = rbclogic.Register( key , createlogic );
    //...
    
    key = std::string("TrackingAlg");
    createlogic = (GenLogic * (*)()) &createTrackingAlg;
    status = rbclogic.Register( key , createlogic );

    key = std::string("SectorORLogic");
    createlogic = (GenLogic * (*)()) &createSectorORLogic;
    status = rbclogic.Register( key , createlogic );
    
    key = std::string("TwoORLogic");
    createlogic = (GenLogic * (*)()) &createTwoORLogic;
    status = rbclogic.Register( key , createlogic );

    key = std::string("WedgeORLogic");
    createlogic = (GenLogic * (*)()) &createWedgeORLogic;
    status = rbclogic.Register( key , createlogic );
    
    key = std::string("PointingLogic");
    createlogic = (GenLogic * (*)()) &createPointingLogic;
    status = rbclogic.Register( key , createlogic );

    assert(status);
    return rbclogic;
    
  };
  
  static RBCLogicType const& rbclogic() {
    static const RBCLogicType s_rbclogic = initialise();
    return s_rbclogic;
  }
  
};
#endif // LOGICTOOL_H
