#include "Alignment/KalmanAlignmentAlgorithm/interface/KalmanAlignmentMetricsCalculator.h"
#include <limits.h>

KalmanAlignmentMetricsCalculator::KalmanAlignmentMetricsCalculator( void ) : theMaxDistance( SHRT_MAX ) {}


KalmanAlignmentMetricsCalculator::~KalmanAlignmentMetricsCalculator( void ) { clear(); }


void KalmanAlignmentMetricsCalculator::updateDistances( const std::vector< Alignable* >& alignables )
{
  // List of the distances of the current alignables.
  FullDistancesList currentDistances;

  // Updated list of the distances of the current alignables.
  FullDistancesList updatedDistances;

  // List of distances between the current and all other alignables that changed due to the update of the list
  // of the current alignables. This information has to be propagated to the lists of all other alignables.
  FullDistancesList propagatedDistances;

  // Iterate over current alignables and set the distances between them to 1 - save pointers to
  // their distances-lists for further manipulation.
  std::vector< Alignable* >::const_iterator itA;
  for ( itA = alignables.begin(); itA != alignables.end(); ++itA )
  {
    FullDistancesList::iterator itD = theDistances.find( *itA );

    if ( itD != theDistances.end() )
    {
      currentDistances[*itA] = itD->second;

      std::vector< Alignable* >::const_iterator itA2;
      for ( itA2 = alignables.begin(); itA2 != alignables.end(); ++itA2 ) (*itD->second)[*itA2] = ( *itA == *itA2 ) ? 0 : 1;
    }
    else
    {
      SingleDistancesList* newEntry = new SingleDistancesList;
      theDistances[*itA] = newEntry;
      currentDistances[*itA] = newEntry;

      std::vector< Alignable* >::const_iterator itA2;
      for ( itA2 = alignables.begin(); itA2 != alignables.end(); ++itA2 ) (*newEntry)[*itA2] = ( *itA == *itA2 ) ? 0 :  1;
    }
  }

  // Iterate over the current alignables' distances-lists and compute updates.
  FullDistancesList::iterator itC1;
  FullDistancesList::iterator itC2;
  for ( itC1 = currentDistances.begin(); itC1 != currentDistances.end(); ++itC1 )
  {
    SingleDistancesList* updatedList = new SingleDistancesList( *itC1->second );

    for ( itC2 = currentDistances.begin(); itC2 != currentDistances.end(); ++itC2 )
    {
      if ( itC1->first != itC2->first ) updateList( updatedList, itC2->second );
    }

    extractPropagatedDistances( propagatedDistances, itC1->first, itC1->second, updatedList );
    updatedDistances[itC1->first] = updatedList;
  }

  // Insert the updated distances-lists.
  insertUpdatedDistances( updatedDistances );

  // Insert the propagated distances-lists.
  insertPropagatedDistances( propagatedDistances );

  // Used only temporary - clear it to deallocate its memory.
  clearDistances( propagatedDistances );
}


const KalmanAlignmentMetricsCalculator::SingleDistancesList&
KalmanAlignmentMetricsCalculator::getDistances( Alignable* i ) const
{
  FullDistancesList::const_iterator itD = theDistances.find( i );
  if ( itD == theDistances.end() ) return theDefaultReturnList;
  return *itD->second;
}


short int KalmanAlignmentMetricsCalculator::operator()( Alignable* i, Alignable* j ) const
{
  if ( i == j ) return 0;

  FullDistancesList::const_iterator itD = theDistances.find( i );
  if ( itD == theDistances.end() ) return -1;

  SingleDistancesList::const_iterator itL = itD->second->find( j );
  if ( itL == itD->second->end() ) return -1;

  return itL->second;
}


unsigned int KalmanAlignmentMetricsCalculator::nDistances( void ) const
{
  unsigned int nod = 0;

  FullDistancesList::const_iterator itD;
  for ( itD = theDistances.begin(); itD != theDistances.end(); ++itD ) nod += itD->second->size();

  return nod;
}


void KalmanAlignmentMetricsCalculator::clear( void )
{
  clearDistances( theDistances );
}


const std::vector< Alignable* > KalmanAlignmentMetricsCalculator::alignables( void ) const
{
  std::vector< Alignable* > alignables;
  alignables.reserve( theDistances.size() );

  for ( FullDistancesList::const_iterator itL = theDistances.begin(); itL != theDistances.end(); ++itL )
    alignables.push_back( itL->first );

  return alignables;
}


void KalmanAlignmentMetricsCalculator::clearDistances( FullDistancesList& dist )
{
  FullDistancesList::iterator itD;
  for ( itD = dist.begin(); itD != dist.end(); ++itD ) delete itD->second;
  dist.clear();
}


void KalmanAlignmentMetricsCalculator::updateList( SingleDistancesList* thisList,
				    SingleDistancesList* otherList )
{
  SingleDistancesList::iterator itThis;
  SingleDistancesList::iterator itOther;

  // Iterate through the ordered entries (via "<") of thisList and otherList.
  for ( itThis = thisList->begin(), itOther = otherList->begin(); itOther != otherList->end(); ++itOther )
  {
    // Irrelevant information.
    if ( itOther->second >= theMaxDistance ) continue;

    // Skip these elements of thisList - no new information available for them in otherList.
    while ( itThis != thisList->end() && itThis->first < itOther->first ) ++itThis;

    // Insert new element ...
    if ( itThis == thisList->end() || itThis->first > itOther->first ) {
      (*thisList)[itOther->first] = itOther->second + 1;
    // ... or write smaller distance for existing element.
    } else if ( itThis->second > itOther->second ) {
      itThis->second = itOther->second + 1;
    }

  }
}


void KalmanAlignmentMetricsCalculator::insertUpdatedDistances( FullDistancesList& updated )
{
  FullDistancesList::iterator itOld;
  FullDistancesList::iterator itNew;

  for ( itNew = updated.begin(); itNew != updated.end(); ++itNew )
  {
    itOld = theDistances.find( itNew->first );
    delete itOld->second;
    itOld->second = itNew->second;
  }
}


void KalmanAlignmentMetricsCalculator::insertPropagatedDistances( FullDistancesList& propagated )
{
  FullDistancesList::iterator itOld;
  FullDistancesList::iterator itNew;

  for ( itNew = propagated.begin(); itNew != propagated.end(); ++itNew )
  {
    itOld = theDistances.find( itNew->first );
    SingleDistancesList::iterator itL;
    for ( itL = itNew->second->begin(); itL != itNew->second->end(); ++itL )
      insertDistance( itOld->second, itL->first, itL->second );
  }
}


void KalmanAlignmentMetricsCalculator::extractPropagatedDistances( FullDistancesList& changes,
						    Alignable* alignable,
						    SingleDistancesList* oldList,
						    SingleDistancesList* newList )
{
  SingleDistancesList::iterator itOld;
  SingleDistancesList::iterator itNew;

  SingleDistancesList newConnections;

  // Distances-list newList has at least entries for the same indices as distances-list
  // oldList. For this reason 'newList->begin()->first <= oldList->begin()->first' and
  // hence 'itNew->first <= itOld->first' is always true in the loop below.
  for ( itOld = oldList->begin(), itNew = newList->begin(); itNew != newList->end(); ++itNew )
  {
    // No entry associated to index itNew->first present in oldList. --> This is indeed a change.
    if ( itOld == oldList->end() || itNew->first < itOld->first )
    {
      insertDistance( changes, itNew->first, alignable, itNew->second );
      newConnections[itNew->first] = itNew->second;
    // Entry associated to index itNew->first present in oldList. --> Check if it has changed.
    } else if ( itNew->first == itOld->first ) {
      if ( itNew->second != itOld->second ) insertDistance( changes, itNew->first, alignable, itNew->second );
      ++itOld;
    }
  }

  SingleDistancesList::iterator it;
  for ( it = newConnections.begin(); it != newConnections.end(); ++it )
    connect( changes, newList, it->first, it->second );
}


void KalmanAlignmentMetricsCalculator::connect( FullDistancesList& changes, SingleDistancesList* connection,
						Alignable* alignable, short int value )
{
  SingleDistancesList::iterator itL;
  for ( itL = connection->begin(); itL != connection->end(); ++itL )
  {
    if ( itL->first != alignable ) insertDistance( changes, alignable, itL->first, value + itL->second );
  }

}


void KalmanAlignmentMetricsCalculator::insertDistance( FullDistancesList& dist, Alignable* i, Alignable* j, short int value )
{
  FullDistancesList::iterator itD = dist.find( i );
  if ( itD != dist.end() ) // Found distances-list for index i.
  {
    insertDistance( itD->second, j, value );
  }
  else // No distances-list found for value i.
  {
    SingleDistancesList* newList = new SingleDistancesList;
    (*newList)[j] = value;
    dist[i] = newList;
  }
}


void KalmanAlignmentMetricsCalculator::insertDistance( SingleDistancesList* distList, Alignable* j, short int value )
{
  SingleDistancesList::iterator itL = distList->find( j );
  if ( itL != distList->end() ) { // Entry associated to index j found.
    if ( itL->second > value ) itL->second = value;
  } else { // No entry associated to index j found. -> Insert new entry.
    (*distList)[j] = value;
  }
}


void KalmanAlignmentMetricsCalculator::writeDistances( std::string filename )
{

}


void KalmanAlignmentMetricsCalculator::readDistances( std::string filename )
{

}


void KalmanAlignmentMetricsCalculator::createBranches( TTree* tree )
{

}


void KalmanAlignmentMetricsCalculator::setBranchAddresses( TTree* tree )
{

}

