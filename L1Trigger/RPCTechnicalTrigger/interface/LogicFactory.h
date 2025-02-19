// $Id: LogicFactory.h,v 1.1 2009/05/16 19:43:30 aosorio Exp $
#ifndef LOGICFACTORY_H 
#define LOGICFACTORY_H 1

// Include files
#include <stdlib.h>
#include <string>
#include <map>

/** @class LogicFactory LogicFactory.h
 *  
 *
 *  @author Andres Osorio
 *
 *  email: aosorio@uniandes.edu.co
 *
 *  @date   2008-10-11
 */

template <class Ilogic, typename Identifier, typename LogicCreator = Ilogic * (*)()> 
class LogicFactory {
public: 
  
  bool Register( const Identifier & id, LogicCreator creator)
  {
    return m_associations.insert(typename std::map<Identifier, LogicCreator>::value_type(id,creator)).second;
  }
  
  bool Unregister( const Identifier & id )
  {
    typename std::map<Identifier, LogicCreator>::const_iterator itr;
    itr = m_associations.find(id);
    if( itr != m_associations.end() ) {
      delete ( itr->second )() ;
    }
    return m_associations.erase(id) == 1;
  }
  
  Ilogic* CreateObject( const Identifier & id )
  {
    typename std::map<Identifier, LogicCreator>::const_iterator itr;
    itr = m_associations.find( id );
    
    if ( itr != m_associations.end() )  {
      return ( itr->second )();
    } else return NULL; // handle error
  }
  
protected:
  
private:
  
  typename std::map<Identifier, LogicCreator> m_associations;
  
};
#endif // LOGICFACTORY_H
