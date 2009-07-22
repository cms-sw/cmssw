// -*- C++ -*-

#if !defined(EventContainer_H)
#define EventContainer_H

#include <map>
#include <string>
#include <typeinfo>

#include "TH1.h"
#include "TFile.h"

#include "DataFormats/FWLite/interface/EventBase.h"
#include "PhysicsTools/FWLite/interface/TH1Store.h"

namespace fwlite
{

   class EventContainer : public fwlite::EventBase
   {
      public:

         //////////////////////
         // Public Constants //
         //////////////////////

         typedef std::map< std::string, std::string >   SSMap;
         typedef void ( *FuncPtr ) (std::string&);

         /////////////
         // friends //
         /////////////
         // tells particle data how to print itself out
         friend std::ostream& operator<< (std::ostream& o_stream, 
                                          const EventContainer &rhs);

         //////////////////////////
         //            _         //
         // |\/|      |_         //
         // |  |EMBER | UNCTIONS //
         //                      //
         //////////////////////////

         /////////////////////////////////
         // Constructors and Destructor //
         /////////////////////////////////
         EventContainer (FuncPtr funcPtr);
         ~EventContainer();

         ////////////////
         // One Liners //
         ////////////////

         // return number of events seen
         int eventsSeen () const { return m_eventsSeen; }

         //////////////////////////////
         // Regular Member Functions //
         //////////////////////////////

         // adds a histogram pointer to the map
         void add (TH1 *histPtr);

         // given a string, returns corresponding histogram pointer
         TH1* hist (const std::string &name);
         TH1* hist (const char* name)
         { return hist( (const std::string) name); }
         TH1* hist (const TString &name)
         { return hist( (const char*) name ); }

         ///////////////////////////////////////////////////////////////////
         // Implement the two functions needed to make this an EventBase. //
         ///////////////////////////////////////////////////////////////////
         bool getByLabel (const std::type_info& iInfo,
                          const char* iModuleLabel,
                          const char* iProductInstanceLabel,
                          const char* iProcessLabel,
                          void* oData) const;

         const std::string getBranchNameFor (const std::type_info& iInfo,
                                             const char* iModuleLabel,
                                             const char* iProductInstanceLabel,
                                             const char* iProcessLabel) const;

         const EventContainer& operator++();

         const EventContainer& toBegin();

         bool atEnd() const;
      
         /////////////////////////////
         // Static Member Functions //
         /////////////////////////////


      private:

         //////////////////////////////
         // Private Member Functions //
         //////////////////////////////

         // stop the copy constructor
         EventContainer (const EventContainer &rhs) {}

         /////////////////////////
         // Private Member Data //
         /////////////////////////

         fwlite::EventBase *m_eventBasePtr;
         TH1Store           m_histStore;
         std::string        m_outputName;
         int                m_eventsSeen;
         int                m_maxWanted;
         int                m_outputEvery;

         ////////////////////////////////
         // Private Static Member Data //
         ////////////////////////////////

         static bool sm_autoloaderCalled;

   };
}


#endif // EventContainer_H
