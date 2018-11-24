#ifndef LOGICFACTORY_H 
#define LOGICFACTORY_H 1

// Include files
#include <cstdlib>
#include <string>
#include <map>
#include <memory>
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
    return m_associations.erase(id) == 1;
  }
  
  std::unique_ptr<Ilogic> CreateObject( const Identifier & id ) const
  {
    auto itr = m_associations.find( id );
    
    if ( itr != m_associations.end() )  {
      return std::unique_ptr<Ilogic>{( itr->second )()};
    } else return nullptr; // handle error
  }
  
protected:
  
private:
  
  typename std::map<Identifier, LogicCreator> m_associations;
  
};
#endif // LOGICFACTORY_H
