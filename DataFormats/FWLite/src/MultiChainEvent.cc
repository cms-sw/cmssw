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
// $Id: MultiChainEvent.cc,v 1.5 2009/08/18 17:56:59 chrjones Exp $
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
				   const std::vector<std::string>& iFileNames2)
{
  event1_ = boost::shared_ptr<ChainEvent> ( new ChainEvent( iFileNames1 ) );
  event2_ = boost::shared_ptr<ChainEvent> ( new ChainEvent( iFileNames2 ) );

  getter_ = boost::shared_ptr<internal::MultiProductGetter>( new internal::MultiProductGetter( this) );

  event1_->setGetter( getter_ );
  event2_->setGetter( getter_ );
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
const MultiChainEvent& 
MultiChainEvent::to(Long64_t iIndex) {
  event1_->to( iIndex );
  return *this;
}


///Go to event with event id "id"
const MultiChainEvent& 
MultiChainEvent::to(edm::EventID id) {
  return to(id.run(), id.event());
}

///Go to event with given run and event number
const MultiChainEvent& 
MultiChainEvent::to(edm::RunNumber_t run, edm::EventNumber_t event) {
  event1_->to( run, event );
  return *this;
}


///Go to the event at index iIndex
const MultiChainEvent& 
MultiChainEvent::toSec(Long64_t iIndex) {
  event2_->to( iIndex );
  return *this;
}


///Go to event with event id "id"
const MultiChainEvent& 
MultiChainEvent::toSec(edm::EventID id) {
  return toSec(id.run(), id.event());
}

///Go to event with given run and event number
const MultiChainEvent& 
MultiChainEvent::toSec(edm::RunNumber_t run, edm::EventNumber_t event) {
  event2_->to( run, event );
  return *this;
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
    event2_->to( event1_->id() );
    bool ret2 = event2_->getByLabel(iType,iModule,iInstance,iProcess,iValue);
    if ( !ret2 ) return false;
    else return true;
  }
  else return true;
}

edm::EDProduct const* MultiChainEvent::getByProductID(edm::ProductID const&iID) const 
{
  // First try the first file
  edm::EDProduct const * prod = this->primary()->event()->getByProductID(iID);
  // Did not find the product, try again
  if ( 0 == prod ) {
    (const_cast<MultiChainEvent*>(this))->toSec(this->id());
    prod = this->secondary()->event()->getByProductID(iID);
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
