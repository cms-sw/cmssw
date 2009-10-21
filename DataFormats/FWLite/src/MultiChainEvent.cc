// -*- C++ -*-
//
// Package:     FWLite
// Class  :     MultiChainEvent
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Salvatore Rappoccio
//         Created:  Thu Jul  9 22:05:56 CDT 2009
// $Id: MultiChainEvent.cc,v 1.8 2009/10/20 18:34:58 srappocc Exp $
//

// system include files

// user include files
#include "DataFormats/FWLite/interface/MultiChainEvent.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Common/interface/EDProductGetter.h"
#include "TFile.h"
#include "TTree.h"

namespace fwlite {

  namespace internal {
    
    class MultiProductGetter : public edm::EDProductGetter {
public:
      MultiProductGetter(MultiChainEvent const * iEvent) : event_(iEvent) {}
      
      virtual edm::EDProduct const*
      getIt(edm::ProductID const& iID) const {

	return event_->getByProductID( iID );
      }
private:
      MultiChainEvent const * event_;
      
    };
  }


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
  MultiChainEvent::MultiChainEvent(const std::vector<std::string>& iFileNames1,
				   const std::vector<std::string>& iFileNames2,
				   bool useSecFileMapSorted)
{
  event1_ = boost::shared_ptr<ChainEvent> ( new ChainEvent( iFileNames1 ) );
  event2_ = boost::shared_ptr<ChainEvent> ( new ChainEvent( iFileNames2 ) );

  getter_ = boost::shared_ptr<internal::MultiProductGetter>( new internal::MultiProductGetter( this) );

  event1_->setGetter( getter_ );
  event2_->setGetter( getter_ );

  useSecFileMapSorted_ = useSecFileMapSorted;

  if ( !useSecFileMapSorted_ ) {
    std::cout << "------------------------------------------------------------------------" << std::endl;
    std::cout << "WARNING! What you are about to do may be very slow." << std::endl;
    std::cout << "The 2-file solution in FWLite works with very simple assumptions." << std::endl;
    std::cout << "It will linearly search through the files in the secondary file list for Products." << std::endl;
    std::cout << "There are speed improvements available to make this run faster." << std::endl;
    std::cout << "***If your secondary files are sorted with a run-range within a file, (almost always the case) " << std::endl;
    std::cout << "***please use the option useSecFileMapSorted=true in this constructor. " << std::endl;
    std::cout << "    > usage: MultiChainEvent( primaryFiles, secondaryFiles, true);" << std::endl;
    std::cout << "------------------------------------------------------------------------" << std::endl;    
    
  }

  if ( useSecFileMapSorted_ ) {

    std::cout << "------------------------------------------------------------------------" << std::endl;    
    std::cout << "This MultiChainEvent is now creating a (run_range)_2 ---> file_index_2 map" << std::endl;
    std::cout << "for the 2-file solution. " << std::endl;
    std::cout << "This is assuming the files you are giving me are sorted by run,event pairs within each secondary file." << std::endl;
    std::cout << "If this is not true (rarely the case), set this option to false." << std::endl;
    std::cout << "    > usage: MultiChainEvent( primaryFiles, secondaryFiles, false);" << std::endl;
    std::cout << "------------------------------------------------------------------------" << std::endl;    
    // speed up secondary file access with a (run,event)_1 ---> index_2 map
    

    // Loop over events, when a new file is encountered, store the first run number from this file,
    // and the last run number from the last file. 
    TFile * lastFile = 0;
    std::pair<event_id_range,Long64_t> eventRange;
    bool firstFile = true;

    bool foundAny = false;

    for( event2_->toBegin();
	 ! event2_->atEnd();
	 ++(*event2_)) {
      // if we have a new file, cache the "first"
      if ( lastFile != event2_->getTFile() ) {

	// if this is not the first file, we have an entry.
	// Add it to the list.
	if ( !firstFile ) {
	  foundAny = true;
	  event_id_range toAdd = eventRange.first;
	  secFileMapSorted_[ toAdd ] = eventRange.second;
	}
	// always add the "first" event id to the cached event range
	eventRange.first.first = event2_->event()->id();
	lastFile = event2_->getTFile();
      }
      // otherwise, cache the "second" event id in the cached event range.
      // Upon the discovery of a new file, this will be used as the
      // "last" event id in the cached event range. 
      else {
	eventRange.first.second = event2_->event()->id();
	eventRange.second = event2_->eventIndex();
      }
      firstFile = false;
    }
    // due to the invailability of a "look ahead" operation, we have one additional "put" to make
    // after the loop (which puts the "last" event, not "this" event.
    if ( foundAny ) {
      event_id_range toAdd = eventRange.first;
      secFileMapSorted_[ toAdd ] = eventRange.second;
    }
//     std::cout << "Dumping run range to event id list:" << std::endl;
//     for ( sec_file_range_index_map::const_iterator mBegin = secFileMapSorted_.begin(),
// 	    mEnd = secFileMapSorted_.end(),
// 	    mit = mBegin;
// 	  mit != mEnd; ++mit ) {
//       char buff[1000];
//       event2_->to( mit->second );
//       sprintf(buff, "[%10d,%10d - %10d,%10d] ---> %10d",
// 	      mit->first.first.run(),
// 	      mit->first.first.event(),
// 	      mit->first.second.run(),
// 	      mit->first.second.event(),
// 	      mit->second );
//       std::cout << buff << std::endl;
//     }
  }

}

// MultiChainEvent::MultiChainEvent(const MultiChainEvent& rhs)
// {
//    // do actual copying here;
// }

MultiChainEvent::~MultiChainEvent()
{
}

//
// assignment operators
//
// const MultiChainEvent& MultiChainEvent::operator=(const MultiChainEvent& rhs)
// {
//   //An exception safe implementation is
//   MultiChainEvent temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

const MultiChainEvent& 
MultiChainEvent::operator++()
{
   event1_->operator++();
   return *this;
}

///Go to the event at index iIndex
bool
MultiChainEvent::to(Long64_t iIndex) 
{
  return event1_->to( iIndex );
}


///Go to event with event id "id"
bool
MultiChainEvent::to(edm::EventID id) 
{
  return to (id.run(), id.event());
}

///Go to event with given run and event number
bool
MultiChainEvent::to (edm::RunNumber_t run, edm::EventNumber_t event) 
{
   return event1_->to( run, event );
}


///Go to the event at index iIndex
bool
MultiChainEvent::toSec(Long64_t iIndex) 
{
   return event2_->to( iIndex );
}

// Go to event with event id "id"
bool
MultiChainEvent::toSec (const edm::EventID &id) 
{
   // First try this file.
   if ( event2_->event_->to( id ) ) 
   {
      // Foudn it, return. 
      return true;
   }
   // Second, assume that the secondary files are each in run/event
   // order.  So, let's loop over all files and see if we can figure
   // out where the event ought to be.
   for ( sec_file_range_index_map::const_iterator mBegin = 
            secFileMapSorted_.begin(),
            mEnd = secFileMapSorted_.end(),
            mit = mBegin;
         mit != mEnd; 
         ++mit ) 
   {
      if ( id < mit->first.first || id > mit->first.second ) 
      {
         // We don't expect this event to be in this file, so don't
         // bother checking it right now.
         continue;
      }
      // If we're here, then we have a reasonable belief that this
      // event is in this secondary file.  This part is
      // expensive. switchToFile does memory allocations and opens the
      // files which becomes very time consuming. This should be done
      // as infrequently as possible.
      event2_->switchToFile( mit->second );
      // Is it here?
      if (event2_->to( id ))
      {
         // Yes!
         return true;
      }
      // if we assumed that the secondary files were not each in
      // order, but were non-overlapping, we could break here.  But at
      // this point, we might as well keep going.
   } // for loop over files

   // if we are still here, then we did not find the id in question,
   // do it the old fashioned way.  This will open up each secondary
   // file and explicitly check to see if the event is there.
   if (event2_->to(id))
   {
      return true;
   }
   // if we're still here, then there really is no matching event in
   // the secondary files.  Throw.
   throw cms::Exception("ProductNotFound") << "Cannot find id " 
                                           << id.run() << ", " 
                                           << id.event() 
                                           << " in secondary list. Exiting." 
                                           << std::endl;
   // to make the compiler happy
   return false;
}
  
///Go to event with given run and event number
bool 
MultiChainEvent::toSec(edm::RunNumber_t run, edm::EventNumber_t event) 
{
  return toSec( edm::EventID( run, event) );
}

// Go to the very first Event
const MultiChainEvent& 
MultiChainEvent::toBegin() 
{
   event1_->toBegin();
   return *this;
}

//
// const member functions
//
const std::string 
MultiChainEvent::getBranchNameFor(const std::type_info& iType, 
                             const char* iModule, 
                             const char* iInstance, 
                             const char* iProcess) const
{
  return event1_->getBranchNameFor(iType,iModule,iInstance,iProcess);
}

const std::vector<edm::BranchDescription>&
MultiChainEvent::getBranchDescriptions() const
{
  return event1_->getBranchDescriptions();
}

const std::vector<std::string>&
MultiChainEvent::getProcessHistory() const
{
  return event1_->getProcessHistory();
}

edm::EventAuxiliary const&
MultiChainEvent::eventAuxiliary() const
{
   return event1_->eventAuxiliary();
}
   
bool 
MultiChainEvent::getByLabel(const std::type_info& iType, 
                       const char* iModule, 
                       const char* iInstance, 
                       const char* iProcess, 
                       void* iValue) const
{
  bool ret1 = event1_->getByLabel(iType,iModule,iInstance,iProcess,iValue);
  if ( !ret1 ) {
    (const_cast<MultiChainEvent*>(this))->toSec(event1_->id());
    bool ret2 = event2_->getByLabel(iType,iModule,iInstance,iProcess,iValue);
    if ( !ret2 ) return false;
  }
  return true;
}

edm::EDProduct const* MultiChainEvent::getByProductID(edm::ProductID const&iID) const 
{
  // First try the first file
  edm::EDProduct const * prod = event1_->getByProductID(iID);
  // Did not find the product, try secondary file
  if ( 0 == prod ) {
    (const_cast<MultiChainEvent*>(this))->toSec(event1_->id());
    prod = event2_->getByProductID(iID);
    if ( 0 == prod ) {
      throw cms::Exception("ProductNotFound") << "Cannot find product " << iID;
    }
  }
  return prod;
}


bool 
MultiChainEvent::isValid() const
{
  return event1_->isValid();
}
MultiChainEvent::operator bool () const
{
  return *event1_;
}

bool 
MultiChainEvent::atEnd() const 
{
  return event1_->atEnd();
}

Long64_t 
MultiChainEvent::size() const
{
  return event1_->size();
}

TriggerNames const&
MultiChainEvent::triggerNames(edm::TriggerResults const& triggerResults)
{
  TriggerNames const* names = triggerNames_(triggerResults);
  if (names != 0) return *names;

  event1_->fillParameterSetRegistry();
  names = triggerNames_(triggerResults);
  if (names != 0) return *names;

  // If we cannot find it in the primary file, this probably will
  // not help but try anyway
  event2_->to( event1_->id() );
  event2_->fillParameterSetRegistry();
  names = triggerNames_(triggerResults);
  if (names != 0) return *names;

  throw cms::Exception("TriggerNamesNotFound")
    << "TriggerNames not found in ParameterSet registry";
  return *names;
}

//
// static member functions
//
void 
MultiChainEvent::throwProductNotFoundException(const std::type_info& iType, 
                                          const char* iModule,
                                          const char* iInstance,
                                          const char* iProcess) {
  ChainEvent::throwProductNotFoundException(iType,iModule,iInstance,iProcess);
}
}
