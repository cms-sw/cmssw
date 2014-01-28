// This test is adapted from one in the XDAQ source code; here is the
// required copyright notice.

/*************************************************************************
 * XDAQ Components for Distributed Data Acquisition                      *
 * Copyright (C) 2000-2004, CERN.			                 *
 * All rights reserved.                                                  *
 * Authors: J. Gutleber and L. Orsini					 *
 *                                                                       *
 * For the licensing terms see LICENSE.		                         *
 * For the list of contributors see CREDITS.   			         *
 *************************************************************************/

#include "toolbox/mem/HeapAllocator.h"
#include "toolbox/mem/Reference.h"
#include "toolbox/mem/MemoryPoolFactory.h"
#include "toolbox/mem/exception/Exception.h"

#include <cassert>
#include <iostream>

using toolbox::mem::Pool;
using toolbox::mem::MemoryPoolFactory;
using toolbox::mem::getMemoryPoolFactory;
using toolbox::mem::HeapAllocator;
using toolbox::mem::Reference;

void
test_copying_chains(MemoryPoolFactory* g_factory, Pool* g_pool)
{
  // Chain two blocks of 8196 Note that getFrame() is not an accessor;
  // it constructs a new 'frame' to which we are given access through
  // a pointer-to-Reference.
  unsigned const int SIZE = 1;
  Reference* myChain = g_factory->getFrame(g_pool, SIZE);
  myChain->setNextReference(g_factory->getFrame(g_pool, SIZE));

  // Zero copy duplicate chain
  Reference* myChainDuplicate  = myChain->duplicate();

  // The references should be distinct.
  assert(myChainDuplicate!=myChain);

  // The data blocks referenced should be identical, not merely
  // containing equal data.
  assert(myChainDuplicate->getDataLocation()==
                 myChain->getDataLocation());

  // Free both chains

  // The first release does not actually release the frames.
  myChain->release();
  assert(g_pool->getMemoryUsage().getUsed() != 0);

  // The second release should release the frames.
  myChainDuplicate->release();
  assert(g_pool->getMemoryUsage().getUsed() == 0);

  // DO NOT call delete on a Reference* obtained from a call to
  // MemoryPoolFactory::getFrame()!  The memory pool managed by the
  // MemoryPoolFactory works with a finite collection of Reference
  // objects, and recycles them. If you call delete, you will be
  // disposing of something that will be later recycled.  
}


int main()
{
  using namespace toolbox::mem;

  MemoryPoolFactory*  g_factory(getMemoryPoolFactory());
  toolbox::net::URN            g_urn("toolbox-mem-pool","myPool");
  HeapAllocator* g_alloc(new HeapAllocator);
  Pool*          g_pool(g_factory->createPool(g_urn, g_alloc));

  test_copying_chains(g_factory, g_pool);

  // This is not sufficient cleanup; valgrind complains of many leaks.
  g_factory->destroyPool(g_urn);
  g_factory->cleanup();

  delete g_alloc;
}

/// emacs configuration
/// Local Variables: -
/// mode: c++ -
/// c-basic-offset: 2 -
/// indent-tabs-mode: nil -
/// End: -
