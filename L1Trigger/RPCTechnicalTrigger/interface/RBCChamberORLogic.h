// $Id: RBCChamberORLogic.h,v 1.2 2009/06/07 21:18:50 aosorio Exp $
#ifndef RBCCHAMBERORLOGIC_H 
#define RBCCHAMBERORLOGIC_H 1

// Include files

#include "L1Trigger/RPCTechnicalTrigger/interface/RBCLogic.h"
#include "L1Trigger/RPCTechnicalTrigger/interface/RBCInput.h"

#include <iostream>
#include <vector>
#include <map>


/** @class RBCChamberORLogic RBCChamberORLogic.h
 *  
 *  <p>This class works out the mapping of inputs to chambers
 *  and applies a Coincidence Logic. The criteria for coincidence
 *  is set by the parameter m_maxlevel.</p>
 *
 *  @authors <B>Andres Osorio</B>, Flavio Loddo, Marcello Maggi
 *
 *  email: aosorio@uniandes.edu.co
 *
 *
 *  @date   2008-10-11
 */
class RBCChamberORLogic : public RBCLogic {
public: 
  /// Standard constructor
  RBCChamberORLogic( ); 
  
  virtual ~RBCChamberORLogic( ); ///< Destructor
      
  void process ( const RBCInput & , std::bitset<2> & );
  
  void setBoardSpecs( const RBCBoardSpecs::RBCBoardConfig & );
  
  std::bitset<6> * getlayersignal(int _idx) { return & m_layersignal[_idx]; };
  
  typedef std::vector<std::string>::iterator itr2names;
  typedef std::map<std::string,bool>::iterator itr2chambers;
  
  void copymap( const std::bitset<15> & );
  
  void createmap( const std::bitset<15> & );
  
  void reset();
  
  bool evaluateLayerOR( const char *, const char * );
  
  void setmaxlevel( int _mx ) { m_maxlevel = _mx;};
  
  std::bitset<6> * m_layersignal;
  
protected:
  
private:
  
  std::vector<std::string> m_rbname;
  
  std::map<std::string, bool> m_chamber;
  
  int m_maxcb;
  
  int m_maxlevel;
  
};
#endif // RBCCHAMBERORLOGIC_H
