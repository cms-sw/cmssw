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

#endif
