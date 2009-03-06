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
// $Id: ChainEvent.cc,v 1.3 2007/10/30 14:23:17 chrjones Exp $
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
  if(eventIndex_ !=fileNames_.size()-1) {
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
  while(iIndex > accumulatedSize_[offsetIndex] && offsetIndex != accumulatedSize_.size()) {
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
void 
ChainEvent::getByLabel(const std::type_info& iType, 
                       const char* iModule, 
                       const char* iInstance, 
                       const char* iProcess, 
                       void* iValue) const
{
  event_->getByLabel(iType,iModule,iInstance,iProcess,iValue);
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
  if (eventIndex_ == fileNames_.size()-1) {
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
