#ifndef Utilities_PdtEntry_h
#define Utilities_PdtEntry_h
/* \class PdtEntry
 *
 * \author Luca Lista, INFN
 *
 */
#include <string>

class PdtEntry {
public:
  explicit PdtEntry( int pdgId ) : pdgId_( pdgId ) { }
  explicit PdtEntry( const std::string & name ) : pdgId_( 0 ), name_( name ) { }
  int pdgId() const;
  const std::string & name() const;
private:
  int pdgId_;
  std::string name_;
};

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace edm {
  template<>
    inline PdtEntry ParameterSet::getParameter<PdtEntry>(std::string const& name) const {
    const Entry & e = retrieve(name);
    if ( e.typeCode() == 'I' ) 
      return PdtEntry( e.getInt32() );
    else if( e.typeCode() == 'S' ) 
      return PdtEntry( e.getString() );
    else 
      throw Exception(errors::Configuration, "EntryError")
	<< "can not convert representation of " << name 
	<< " to value of type PdtEntry. "
	<< "Please, provide a parameter either of type int32 or string.";
  }
}

#endif
