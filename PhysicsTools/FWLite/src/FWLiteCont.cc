// -*- C++ -*-
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cassert>
#include "FWCore/FWLite/interface/AutoLibraryLoader.h"
#include "PhysicsTools/FWLite/interface/FWLiteCont.h"
#include "PhysicsTools/FWLite/interface/OptionUtils.h"
#include "DataFormats/FWLite/interface/ChainEvent.h"

using namespace std;

////////////////////////////////////
// Static Member Data Declaration //
////////////////////////////////////

bool FWLiteCont::m_autoloaderCalled = false;


FWLiteCont::FWLiteCont (FuncPtr funcPtr) : m_eventsSeen (0), m_maxWanted (0)
{
   // Call the autoloader if not already called.
   if (! m_autoloaderCalled)
   {
      AutoLibraryLoader::enable();
      m_autoloaderCalled = true;      
   }

   // get the user-defined tag
   string tag;
   (*funcPtr) (tag);

   // finish defaultt options
   optutl::_finishDefaultOptions (tag);
   m_eventBasePtr = 
      new fwlite::ChainEvent(optutl::stringVector ("inputFiles") );
   m_outputName = optutl::stringValue ("outputName");

   m_maxWanted   = optutl::integerValue ("maxEvent");
   m_outputEvery = optutl::integerValue ("outputEvery");
}

FWLiteCont::~FWLiteCont()
{
   // if the pointer is non-zero, then we should run the standard
   // destructor.  If it is zero, then we should do nothing
   if (! m_eventBasePtr)
   {
      return;
   } 
   // If we're still here, let's get to work.
   m_histStore.write (m_outputName);
   delete m_eventBasePtr;
}

void
FWLiteCont::add (TH1 *histPtr)
{
   m_histStore.add (histPtr);
}

TH1*
FWLiteCont::hist (const string &name)
{
   return m_histStore.hist (name);
}

bool 
FWLiteCont::getByLabel (const std::type_info& iInfo,
                        const char* iModuleLabel,
                        const char* iProductInstanceLabel,
                        const char* iProcessLabel,
                        void* oData) const
{
   assert (m_eventBasePtr);
   return m_eventBasePtr->getByLabel( iInfo, 
                                      iModuleLabel, 
                                      iProductInstanceLabel,
                                      iProcessLabel, 
                                      oData );
}

const std::string 
FWLiteCont::getBranchNameFor (const std::type_info& iInfo,
                              const char* iModuleLabel,
                              const char* iProductInstanceLabel,
                              const char* iProcessLabel) const
{
   assert (m_eventBasePtr);
   return m_eventBasePtr->getBranchNameFor( iInfo,
                                            iModuleLabel,
                                            iProductInstanceLabel,
                                            iProcessLabel );
}

const FWLiteCont& 
FWLiteCont::operator++()
{
   assert (m_eventBasePtr);

   // What we should do here is put a virtual function in the base
   // class called plusplusOperator.  This function should be called
   // by each derived class' operator++() function.  In this case, I
   // could then:
   //
   // m_eventBasePtr->plusPlusOperator()
   // return *this;

   // Since this isn't setup, I'm going to have to resort to dynamic
   // casting.  Either m_eventBasePtr is a pointer to ChainEvent or
   // (eventually) MultiChainEvent.
   fwlite::ChainEvent *chainEventPtr = 
      dynamic_cast< fwlite::ChainEvent* > ( m_eventBasePtr );
   assert (chainEventPtr);
   chainEventPtr->operator++();
   ++m_eventsSeen;
   if (m_outputEvery && m_eventsSeen % m_outputEvery == 0 ) 
   {
      cout << "Processing Event: " << m_eventsSeen << endl;
   }
   return *this;   
}

const FWLiteCont& 
FWLiteCont::toBegin()
{
   assert (m_eventBasePtr);
   m_eventsSeen = 0;

   // same comment here as in operator++
   fwlite::ChainEvent *chainEventPtr = 
      dynamic_cast< fwlite::ChainEvent* > ( m_eventBasePtr );
   assert (chainEventPtr);
   chainEventPtr->toBegin();

   // If we're going to skip over any events, do it here.

   // O.k.  We should be good to go.
   return *this;
}

bool
FWLiteCont::atEnd() const
{
   // first check to see that we haven't already processed the maxinum
   // number of events that we asked for.
   if (m_maxWanted && m_eventsSeen >= m_maxWanted)
   {
      // we're done
      return true;
   }

   // now let's make sure there are still events.  Same comment here
   // as in operator++.
   fwlite::ChainEvent *chainEventPtr = 
      dynamic_cast< fwlite::ChainEvent* > ( m_eventBasePtr );
   assert (chainEventPtr);
   return chainEventPtr->atEnd();
}


// friends
ostream& operator<< (ostream& o_stream, const FWLiteCont &rhs)
{
   return o_stream;
} 
