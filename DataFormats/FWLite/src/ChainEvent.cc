// -*- C++ -*-
//
// Package:     FWLite
// Class  :     ChainEvent
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Sat Jun 16 06:48:39 EDT 2007
// $Id: ChainEvent.cc,v 1.8 2009/07/12 05:09:09 srappocc Exp $
//

// system include files

// user include files
#include "DataFormats/FWLite/interface/ChainEvent.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "TFile.h"
#include "TTree.h"

namespace fwlite {
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
  ChainEvent::ChainEvent(const std::vector<std::string>& iFileNames):
  fileNames_(iFileNames),
  file_(),
  event_(),
  eventIndex_(0),
  accumulatedSize_()
{
    Long64_t summedSize=0;
    accumulatedSize_.reserve(iFileNames.size());
    for(std::vector<std::string>::const_iterator it= iFileNames.begin(),
        itEnd = iFileNames.end();
        it!=itEnd;
        ++it) {
      file_ = boost::shared_ptr<TFile>(TFile::Open(it->c_str()));
      TTree* tree = dynamic_cast<TTree*>(file_->Get(edm::poolNames::eventTreeName().c_str()));
      if(0==tree) {
        throw cms::Exception("NotEdmFile")<<"The file "<<*it<<" has no 'Events' TTree and therefore is not an EDM ROOT file";
      }
      
      summedSize += tree->GetEntries();
      accumulatedSize_.push_back(summedSize);
    }
    if ( iFileNames.size() > 0 ) 
      switchToFile(0);
}

// ChainEvent::ChainEvent(const ChainEvent& rhs)
// {
//    // do actual copying here;
// }

ChainEvent::~ChainEvent()
{
}

//
// assignment operators
//
// const ChainEvent& ChainEvent::operator=(const ChainEvent& rhs)
// {
//   //An exception safe implementation is
//   ChainEvent temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
const ChainEvent& 
ChainEvent::operator++()
{
  if(eventIndex_ != static_cast<Long64_t>(fileNames_.size())-1) {
    ++(*event_);
    if(event_->atEnd()) {
      switchToFile(++eventIndex_);
    }
  } else {
    if(*event_) {
      ++(*event_);
    }
  } 
  return *this;
}

///Go to the event at index iIndex
const ChainEvent& 
ChainEvent::to(Long64_t iIndex) {
  if(iIndex > accumulatedSize_.back()) {
    //should throw exception
    return *this;
  }
  Long64_t offsetIndex = eventIndex_;
  bool incremented = false;
  while(iIndex > accumulatedSize_[offsetIndex] && offsetIndex != static_cast<Long64_t>(accumulatedSize_.size())) {
    ++offsetIndex;
    incremented = true;
  }
  if (incremented) {
    //we will have over shot
    --offsetIndex;
  } else {
    while(iIndex < accumulatedSize_[offsetIndex] && offsetIndex !=-1) {
      --offsetIndex;
    }
  }
  if(offsetIndex-1 != eventIndex_) {
    switchToFile(eventIndex_ = offsetIndex+1);
  }
  event_->to( iIndex - accumulatedSize_[offsetIndex]);
  return *this;
}


///Go to event with event id "id"
const ChainEvent& 
ChainEvent::to(edm::EventID id) {
  return to(id.run(), id.event());
}

///Go to event with given run and event number
const ChainEvent& 
ChainEvent::to(edm::RunNumber_t run, edm::EventNumber_t event) {

  // First try this file
  if ( event_->to( run, event ) )  {
    // found it, return
    return *this;
  } 
  else {
    // Did not find it, try the other files sequentially.
    // Someday I can make this smarter. For now... we get something working. 
    Long64_t thisFile = eventIndex_;
    std::vector<std::string>::const_iterator filesBegin = fileNames_.begin(),
      filesEnd = fileNames_.end(), ifile = filesBegin;
    for ( ; ifile != filesEnd; ++ifile ) {
      // skip the "first" file that we tried
      if ( ifile - filesBegin != thisFile ) {
	// switch to the next file
	switchToFile( ifile - filesBegin );
	// check that tree for the desired event
	if ( event_->to( run, event ) ) {
	  // found it, return
	  return *this;
	} 
      }// end ignore "first" file that we tried
    }// end loop over files
    
    // did not find the event with id "id".
    // throw exception. 
    throw cms::Exception("EventNotFound")
      << "The ChainEvent does not contain run " 
      << run << ", event " << event;
    return *this; // make the compiler happy. this will not execute
  }// end if we did not find event id in "first" file

}

/** Go to the very first Event*/
const ChainEvent& 
ChainEvent::toBegin() {
  if(eventIndex_ != 0) {
    switchToFile(0);
  }
  event_->toBegin();
  return *this;
}

void
ChainEvent::switchToFile(Long64_t iIndex)
{
  eventIndex_= iIndex;
  file_ = boost::shared_ptr<TFile>(TFile::Open(fileNames_[iIndex].c_str()));
  event_ = boost::shared_ptr<Event>( new Event(file_.get()));
}
//
// const member functions
//
const std::string 
ChainEvent::getBranchNameFor(const std::type_info& iType, 
                             const char* iModule, 
                             const char* iInstance, 
                             const char* iProcess) const
{
  return event_->getBranchNameFor(iType,iModule,iInstance,iProcess);
}

const std::vector<edm::BranchDescription>&
ChainEvent::getBranchDescriptions() const
{
  return event_->getBranchDescriptions();
}

const std::vector<std::string>&
ChainEvent::getProcessHistory() const
{
  return event_->getProcessHistory();
}

edm::EventID
ChainEvent::id() const
{
  return event_->id();
}

const edm::Timestamp&
ChainEvent::time() const
{
  return event_->time();
}

bool 
ChainEvent::getByLabel(const std::type_info& iType, 
                       const char* iModule, 
                       const char* iInstance, 
                       const char* iProcess, 
                       void* iValue) const
{
  return event_->getByLabel(iType,iModule,iInstance,iProcess,iValue);
}

edm::EDProduct const* ChainEvent::getByProductID(edm::ProductID const& iID) const
{
  return event_->getByProductID( iID );
}

bool 
ChainEvent::isValid() const
{
  return event_->isValid();
}
ChainEvent::operator bool () const
{
  return *event_;
}

bool 
ChainEvent::atEnd() const 
{
  if (eventIndex_ == static_cast<Long64_t>(fileNames_.size())-1) {
    return event_->atEnd();
  }
  return false;
}

Long64_t 
ChainEvent::size() const
{
  return accumulatedSize_.back();
}


//
// static member functions
//
void 
ChainEvent::throwProductNotFoundException(const std::type_info& iType, 
                                          const char* iModule,
                                          const char* iInstance,
                                          const char* iProcess) {
  Event::throwProductNotFoundException(iType,iModule,iInstance,iProcess);
}
}
