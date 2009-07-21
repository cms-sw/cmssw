// -*- C++ -*-
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cassert>
#include "PhysicsTools/FWLite/interface/FWLiteCont.h"
#include "PhysicsTools/FWLite/interface/OptionUtils.h"

using namespace std;

////////////////////////////////////
// Static Member Data Declaration //
////////////////////////////////////


FWLiteCont::FWLiteCont() : m_eventBasePtr (0)
{
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
}

void
FWLiteCont::add (TH1 *histPtr)
{
   m_histStore.add (histPtr);
}

TH1*
FWLiteCont::hist (const string &name)
{
   m_histStore.hist (name);
}

void
FWLiteCont::write (const string &filename) const
{
   m_histStore.write (filename);
}

void
FWLiteCont::write (TFile *filePtr) const
{
   m_histStore.write (filePtr);
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
FWLitecont::operator++()
{
   assert (m_eventBasePtr);
   m_eventBasePtr->operator++();
   return *this;
}



// friends
ostream& operator<< (ostream& o_stream, const FWLiteCont &rhs)
{
   return o_stream;
} 
