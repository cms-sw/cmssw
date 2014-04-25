#ifndef EvFMicroStateService_H
#define EvFMicroStateServiceH 1


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
//#include "FWCore/ServiceRegistry/interface/StreamContext.h"

//#include "boost/thread/thread.hpp"

#include <string>
#include <vector>

namespace evf{



    class MicroStateService
    {
    public:
      enum Microstate { mInvalid = 0, mFwkOvh, mIdle, mInput, mInputDone, mDqm, mEoL, mCOUNT}; 
      // the names of the states - some of them are never reached in an online app
      static const edm::ModuleDescription reservedMicroStateNames[mCOUNT];
      MicroStateService(const edm::ParameterSet&,edm::ActivityRegistry&);
      virtual ~MicroStateService();
      
      virtual std::string getMicroState1(){return default_return_;}
      
      virtual std::string const &getMicroState2(){return default_return_;}

      virtual void setMicroState(Microstate m)=0;
      virtual void setMicroState(edm::StreamID sid, Microstate m) = 0;
      
    protected:
      static const std::string default_return_;
      //boost::mutex lock_;
    };

}

#endif
