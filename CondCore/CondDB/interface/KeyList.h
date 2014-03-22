#ifndef CondCore_CondDB_KeyList_h
#define CondCore_CondDB_KeyList_h

#include "CondCore/CondDB/interface/IOVProxy.h"
#include "CondCore/CondDB/interface/Binary.h"
#include "CondCore/CondDB/interface/Serialization.h"
#include "CondCore/CondDB/interface/Exception.h"
#include "CondFormats/Common/interface/BaseKeyed.h"
//
#include<map>
#include<vector>
#include<string>

/*
 * KeyList represents a set of payloads each identified by a key  and "valid" at given time
 * Usually these payloads are configuration objects loaded in anvance
 * The model used here calls for all payloads to be "stored" in a single IOVSequence each identified by a unique key 
 * (properly hashed to be mapped in 64bits)
 *
 * the keylist is just a vector of the hashes each corresponding to a key
 * the correspondence position in the vector user-friendly name is kept in 
 * a list of all "names" that is defined in advance and kept in a dictionary at IOVSequence level
 
 *
 */

namespace cond {

  namespace persistency {

    class KeyList {
    public:
      
      void init( IOVProxy iovProxy );
      
      void load( const std::vector<unsigned long long>& keys );
      
      template<typename T> 
      boost::shared_ptr<T> get(size_t n) const {
	if( n> (size()-1) ) throwException( "Index outside the bounds of the key array.",
					    "KeyList::get");
	if( !m_objects[n] ){
	  auto i = m_data.find( n );
	  if( i != m_data.end() ){
	    m_objects[n] = deserialize<T>( i->second.first, i->second.second.first, i->second.second.second, m_isOra );
	    m_data.erase( n );
	  } else {
	    throwException( "Payload for index "+boost::lexical_cast<std::string>(n)+" has not been found.",
			    "KeyList::get");
	  }
	}
	return boost::static_pointer_cast<T>( m_objects[n] );
      }

      int size() const { return m_objects.size();}

    private:
      // the db session
      IOVProxy m_proxy;
      // the key selection: 
      mutable std::map<size_t,std::pair<std::string,std::pair<cond::Binary,cond::Binary> > > m_data;
      mutable std::vector<boost::shared_ptr<void> > m_objects;
      bool m_isOra = false;
      
    };

  }
}

#endif
