#include "FWCore/MessageLogger/interface/MessageSender.h"
#include "FWCore/MessageLogger/interface/MessageLoggerQ.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"

#include "FWCore/MessageLogger/interface/ErrorSummaryEntry.h"

#include <cassert>
#include <vector>
#include <limits>
#include <atomic>

#include <functional>
#include "tbb/concurrent_unordered_map.h"


#define TRACE_DROP
#ifdef TRACE_DROP
#include <iostream>
#endif

// Change log
//
//  1  mf 8/25/08	keeping the error summary information for
//			LoggedErrorsSummary()
//			
// 2  mf 11/2/10	Use new moduleContext method of MessageDrop:
//			see MessageServer/src/MessageLogger.cc change 17.
//			


using namespace edm;

namespace  {
  //Helper class used as 'key' to the thread safe map storing the
  // per event log error and log warning messages
  struct ErrorSummaryMapKey {
    std::string     category;
    std::string     module;
    ELseverityLevel severity;

    bool operator<(ErrorSummaryMapKey const& iOther) const {
      int comp =severity.getLevel()-iOther.severity.getLevel();
      if(0==comp) {
        comp = category.compare(iOther.category);
        if(comp == 0) {
          comp = module.compare(iOther.module);
        }
      }
      return comp < 0;
    }

    bool operator==(ErrorSummaryMapKey const& iOther) const {
      return ((0==category.compare(iOther.category)) and
              (0==module.compare(iOther.module)) and
              (severity.getLevel() ==iOther.severity.getLevel()));
    }
    size_t smallHash() const {
      std::hash<std::string> h;

      return h( category+module+severity.getSymbol());
    }

    struct key_hash {
      std::size_t operator()(ErrorSummaryMapKey const& iKey) const{
        return iKey.smallHash();
      }
    };
  };
  
  class AtomicUnsignedInt {
  public:
    AtomicUnsignedInt() : value_(0) {}
    AtomicUnsignedInt(AtomicUnsignedInt const& r) : value_(r.value_.load(std::memory_order_acquire)) {}
    std::atomic<unsigned int>& value() { return value_; }
    std::atomic<unsigned int> const& value() const { return value_; }
  private:
    std::atomic<unsigned int> value_;
  };

}

[[cms::thread_safe]] static std::atomic<bool> errorSummaryIsBeingKept{false};
//Each item in the vector is reserved for a different Stream
[[cms::thread_safe]] static std::vector<tbb::concurrent_unordered_map<ErrorSummaryMapKey, AtomicUnsignedInt,ErrorSummaryMapKey::key_hash>> errorSummaryMaps;

MessageSender::MessageSender( ELseverityLevel const & sev, 
			      ELstring const & id,
			      bool verbatim, bool suppressed )
: errorobj_p( suppressed ? 0 : new ErrorObj(sev,id,verbatim), ErrorObjDeleter())
{
  //std::cout << "MessageSender ctor; new ErrorObj at: " << errorobj_p << '\n';
}


// This destructor must not be permitted to throw. A
// boost::thread_resoruce_error is thrown at static destruction time,
// if the MessageLogger library is loaded -- even if it is not used.
void MessageSender::ErrorObjDeleter::operator()(ErrorObj * errorObjPtr) {
  if (errorObjPtr == 0) {
    return;
  }
  try 
    {
      //std::cout << "MessageSender dtor; ErrorObj at: " << errorobj_p << '\n';

      // surrender ownership of our ErrorObj, transferring ownership
      // (via the intermediate MessageLoggerQ) to the MessageLoggerScribe
      // that will (a) route the message text to its destination(s)
      // and will then (b) dispose of the ErrorObj
      
      MessageDrop * drop = MessageDrop::instance();
      if (drop) {
	errorObjPtr->setModule(drop->moduleContext());		// change log 
	errorObjPtr->setContext(drop->runEvent);
      } 
#ifdef TRACE_DROP
      if (!drop) std::cerr << "MessageSender::~MessageSender() - Null drop pointer \n";
#endif
								// change log 1
      if ( errorSummaryIsBeingKept.load(std::memory_order_acquire) &&
           errorObjPtr->xid().severity >= ELwarning &&
          drop->streamID < std::numeric_limits<unsigned int>::max())
      {
        auto& errorSummaryMap =errorSummaryMaps[drop->streamID];

        ELextendedID const & xid = errorObjPtr->xid();
        ErrorSummaryMapKey key {xid.id, xid.module, xid.severity};
        auto i = errorSummaryMap.find(key);
        if (i != errorSummaryMap.end()) {
          i->second.value().fetch_add(1,std::memory_order_acq_rel);  // same as ++errorSummaryMap[key]
        } else {
          errorSummaryMap[key].value().store(1,std::memory_order_release);
        }
      }
      
      MessageLoggerQ::MLqLOG(errorObjPtr);
    }
  catch ( ... )
    {
      // nothing to do
      
      // for test that removal of thread-involved static works, 
      // simply throw here, then run in trivial_main in totalview
      // and Next or Step so that the exception would be detected.
      // That test has been done 12/14/07.
    }
}
MessageSender::~MessageSender()
{
}

//The following functions are declared here rather than in
// LoggedErrorsSummary.cc because only  MessageSender and these
// functions interact with the statics errorSummaryIsBeingKept and
// errorSummaryMaps. By putting them into the same .cc file the
// statics can be file scoped rather than class scoped and therefore
// better encapsulated.
namespace edm {
  
  bool EnableLoggedErrorsSummary() {
    bool ret = errorSummaryIsBeingKept.exchange(true,std::memory_order_acq_rel);
    return ret;
  }
  
  bool DisableLoggedErrorsSummary(){
    bool ret = errorSummaryIsBeingKept.exchange(false,std::memory_order_acq_rel);
    return ret;
  }
  
  bool FreshErrorsExist(unsigned int iStreamID) {
    assert(iStreamID<errorSummaryMaps.size());
    return  errorSummaryMaps[iStreamID].size()>0;
  }
  
  std::vector<ErrorSummaryEntry> LoggedErrorsSummary(unsigned int iStreamID) {
    assert(iStreamID<errorSummaryMaps.size());
    auto const& errorSummaryMap =errorSummaryMaps[iStreamID];
    std::vector<ErrorSummaryEntry> v;
    auto end = errorSummaryMap.end();
    for (auto i = errorSummaryMap.begin();
         i != end; ++i) {
      auto const& key = i->first;
      ErrorSummaryEntry e{key.category,key.module,key.severity};

      e.count = i->second.value().load(std::memory_order_acquire);
      v.push_back(e);
    }
    std::sort(v.begin(),v.end());
    return v;
  }
  
  void clearLoggedErrorsSummary(unsigned int iStreamID) {
    assert(iStreamID<errorSummaryMaps.size());
    errorSummaryMaps[iStreamID].clear();
  }
  
  void setMaxLoggedErrorsSummaryIndicies(unsigned int iMax) {
    assert(0==errorSummaryMaps.size());
    errorSummaryMaps.resize(iMax);
  }

  
  std::vector<ErrorSummaryEntry> LoggedErrorsOnlySummary(unsigned int iStreamID) {    //  ChangeLog 2
    std::vector<ErrorSummaryEntry> v;
    assert(iStreamID < errorSummaryMaps.size());
    auto const& errorSummaryMap =errorSummaryMaps[iStreamID];
    auto end = errorSummaryMap.end();
    for (auto i = errorSummaryMap.begin();
         i != end; ++i) {
      auto const& key = i->first;
      if (key.severity >= edm::ELerror) {
        ErrorSummaryEntry e{key.category,key.module,key.severity};
        e.count = i->second.value().load(std::memory_order_acquire);
        v.push_back(e);
      }
    }
    std::sort(v.begin(),v.end());
    return v;
  }
  
} // end namespace edm

