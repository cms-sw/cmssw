// $Id: FragmentStore.cc,v 1.10 2011/03/30 15:16:48 mommsen Exp $
/// @file: FragmentStore.cc

#include "EventFilter/StorageManager/interface/FragmentStore.h"
#include "EventFilter/StorageManager/interface/Utils.h"

using namespace stor;


FragmentStore::FragmentStore(size_t maxMemoryUsageMB)
  : maxMemoryUsage_(maxMemoryUsageMB*1024*1024)
{
  clear();
}

const bool FragmentStore::addFragment(I2OChain &chain)
{
  // the trivial case that the chain is already complete
  if ( chain.complete() ) return true;

  memoryUsed_ += chain.memoryUsed();
  const FragKey newKey = chain.fragmentKey();

  // Use efficientAddOrUpdates pattern suggested by Item 24 of 
  // 'Effective STL' by Scott Meyers
  fragmentMap::iterator pos = store_.lower_bound(newKey);

  if(pos != store_.end() && !(store_.key_comp()(newKey, pos->first)))
  {
    // key already exists
    pos->second.addToChain(chain);

    if ( pos->second.complete() )
    {
      chain = pos->second;
      store_.erase(pos);
      memoryUsed_ -= chain.memoryUsed();
      return true;
    }
  }
  else
  {
    chain.resetStaleWindowStartTime();

    // The key does not exist in the map, add it to the map
    // Use pos as a hint to insert, so it can avoid another lookup
    store_.insert(pos, fragmentMap::value_type(newKey, chain));
    chain.release();

    // We already handled the trivial case that the chain is complete.
    // Thus, store_ will not have a complete event.
  }

  return false;
}

void FragmentStore::addToStaleEventTimes(const utils::Duration_t duration)
{
  for (
    fragmentMap::iterator it = store_.begin(), itEnd = store_.end();
    it != itEnd;
    ++it
  )
  {
    it->second.addToStaleWindowStartTime(duration);
  }
}

void FragmentStore::resetStaleEventTimes()
{
  for (
    fragmentMap::iterator it = store_.begin(), itEnd = store_.end();
    it != itEnd;
    ++it
  )
  {
    it->second.resetStaleWindowStartTime();
  }
}

const bool FragmentStore::getStaleEvent(I2OChain &chain, utils::Duration_t timeout)
{
  const utils::TimePoint_t cutOffTime = utils::getCurrentTime() - timeout;
  
  fragmentMap::iterator pos = store_.begin();
  fragmentMap::iterator end = store_.end();

  while ( (pos != end) && (pos->second.staleWindowStartTime() > cutOffTime ) )
  {
    ++pos;
  }

  if ( pos == end )
  {
    chain.release();
    return false;
  }
  else
  {
    chain = pos->second;
    store_.erase(pos);
    memoryUsed_ -= chain.memoryUsed();
    chain.markFaulty();
    return true;
  }
}



/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
