// -*- C++ -*-
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cassert>

#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "PhysicsTools/FWLite/interface/EventContainer.h"
#include "PhysicsTools/FWLite/interface/dout.h"
#include "DataFormats/FWLite/interface/MultiChainEvent.h"

#include "TH1.h"

using namespace std;
using namespace fwlite;

////////////////////////////////////
// Static Member Data Declaration //
////////////////////////////////////

bool EventContainer::sm_autoloaderCalled = false;


EventContainer::EventContainer (optutl::CommandLineParser &parser, 
                                FuncPtr funcPtr) : 
   m_eventsSeen (0), m_maxWanted (0), m_parserPtr (0)
{
   // get the user-defined tag
   string tag;
   if (funcPtr)
   {
      (*funcPtr) (tag);
   }

   // finish defaultt options and create fwlite::Event
   parser._finishDefaultOptions (tag);

   // Call the autoloader if not already called.
   if (! sm_autoloaderCalled)
   {
      FWLiteEnabler::enable();
      sm_autoloaderCalled = true;      
   }

   const optutl::CommandLineParser::SVec &secondaryInputFiles = 
      parser.stringVector ("secondaryInputFiles");
   if (secondaryInputFiles.size())
   {
      m_eventBasePtr = 
         new fwlite::MultiChainEvent( parser.stringVector ("inputFiles"), 
                                      secondaryInputFiles,
				      parser.boolValue("orderedSecondaryFiles") );
   } else {
      m_eventBasePtr = 
         new fwlite::ChainEvent( parser.stringVector ("inputFiles") );
   }

   // get whatever other info you want
   m_outputName  = parser.stringValue  ("outputFile");
   m_maxWanted   = parser.integerValue ("maxEvents");
   m_outputEvery = parser.integerValue ("outputEvery");

   // remember my parser
   m_parserPtr = &parser;

   // TH1::AddDirectory(false);
}

EventContainer::~EventContainer()
{
   // if the pointer is non-zero, then we should run the standard
   // destructor.  If it is zero, then we should do nothing
   if (! m_eventBasePtr)
   {
      return;
   } 
   // If we're still here, let's get to work.
   cout << "EventContainer Summary: Processed "
        << m_eventsSeen << " events." << endl;
   optutl::CommandLineParser &parser = this->parser();
   if (optutl::CommandLineParser::kStringVector == 
       parser.hasOption("inputFiles"))
   {
      m_histStore.write (m_outputName, 
                         parser.argVec(),
                         parser.stringVector ("inputFiles"));
   } else {
      m_histStore.write (m_outputName, 
                         parser.argVec());
   }
   delete m_eventBasePtr;
}

void
EventContainer::add (TH1 *histPtr, const string &directory)
{
   m_histStore.add (histPtr, directory);
}

optutl::CommandLineParser &
EventContainer::parser()
{
   assert (m_parserPtr);
   return *m_parserPtr;
}

TH1*
EventContainer::hist (const string &name)
{
   return m_histStore.hist (name);
}

bool 
EventContainer::getByLabel (const std::type_info& iInfo,
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
EventContainer::getBranchNameFor (const std::type_info& iInfo,
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

const EventContainer& 
EventContainer::operator++()
{
   assert (m_eventBasePtr);

   m_eventBasePtr->operator++();
   ++m_eventsSeen;
   if (m_outputEvery && m_eventsSeen % m_outputEvery == 0 ) 
   {
      cout << "Processing Event: " << m_eventsSeen << endl;
   }
   return *this;   
}

const EventContainer& 
EventContainer::toBegin()
{
   assert (m_eventBasePtr);
   m_eventsSeen = 0;
   m_eventBasePtr->toBegin();

   // If we're going to skip over any events, do it here.

   // O.k.  We should be good to go.
   return *this;
}

bool
EventContainer::atEnd() const
{
   assert (m_eventBasePtr);
   // first check to see that we haven't already processed the maxinum
   // number of events that we asked for.
   if (m_maxWanted && m_eventsSeen >= m_maxWanted)
   {
      // we're done
      return true;
   }

   return m_eventBasePtr->atEnd();
}


// friends
ostream& operator<< (ostream& o_stream, const EventContainer &rhs)
{
   return o_stream;
} 
