#include "CondFormats/CSCObjects/interface/CSCBadChambers.h"
#include "DataFormats/MuonDetId/interface/CSCIndexer.h"
#include <algorithm>

bool CSCBadChambers::isInBadChamber( const CSCDetId& id ) const {

  if ( numberOfChambers() == 0 ) return false;

  short int iri = id.ring();
  //@@ Beware future ME11 changes
  if ( iri == 4 ) iri = 1; // reset ME1A to ME11
  CSCIndexer indexer;
  int ilin = indexer.chamberIndex( id.endcap(), id.station(), iri, id.chamber() );
  std::vector<int>::const_iterator badbegin = chambers.begin();
  std::vector<int>::const_iterator badend   = chambers.end();
  std::vector<int>::const_iterator it = std::find( badbegin, badend, ilin );
  if ( it != badend ) return true; // id is in the list of bad chambers
  else return false;
}

