//
// $Id: TriggerPath.cc,v 1.4 2010/04/20 21:39:47 vadler Exp $
//


#include "DataFormats/PatCandidates/interface/TriggerPath.h"


using namespace pat;


/// default constructor

TriggerPath::TriggerPath() :
  name_(),
  index_(),
  prescale_(),
  run_(),
  accept_(),
  error_(),
  lastActiveFilterSlot_()
{
  modules_.clear();
  filterIndices_.clear();
}

/// constructor from values

TriggerPath::TriggerPath( const std::string & name ) :
  name_( name ),
  index_(),
  prescale_(),
  run_(),
  accept_(),
  error_(),
  lastActiveFilterSlot_()
{
  modules_.clear();
  filterIndices_.clear();
}

TriggerPath::TriggerPath( const std::string & name, unsigned index, unsigned prescale, bool run, bool accept, bool error, unsigned lastActiveFilterSlot ) :
  name_( name ),
  index_( index ),
  prescale_( prescale ),
  run_( run ),
  accept_( accept ),
  error_( error ),
  lastActiveFilterSlot_( lastActiveFilterSlot )
{
  modules_.clear();
  filterIndices_.clear();
}

/// getters

// returns size of modules_ if name unknown and -1 if 'modules_' not filled
int TriggerPath::indexModule( const std::string & name ) const
{
  size_t i( 0 );
  while ( i < modules().size() ) {
    if ( name == modules().at( i ) ) {
      return ( int )i;
    }
    ++i;
  }
  return i == 0 ? -1 : ( int )i;
}

std::vector< std::string > TriggerPath::l1Seeds( const bool decision ) const
{

  std::vector< std::string > seeds;
  for ( L1SeedCollection::const_iterator iSeed = l1Seeds().begin(); iSeed != l1Seeds().end(); ++iSeed ) {
    if ( iSeed->first == decision ) seeds.push_back( iSeed->second );
//   for ( size_t iSeed = 0; iSeed < l1Seeds().size(); ++iSeed ) {
//     if ( l1Seeds().at( iSeed ).first == decision ) seeds.push_back( l1Seeds().at( iSeed ).second );
  }
  return seeds;

}
