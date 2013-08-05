#ifndef FWCore_Framework_stream_EDProducerAdaptorBase_h
#define FWCore_Framework_stream_EDProducerAdaptorBase_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     EDProducerAdaptorBase
// 
/**\class edm::stream::EDProducerAdaptorBase EDProducerAdaptorBase.h "FWCore/Framework/interface/stream/EDProducerAdaptorBase.h"

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 02 Aug 2013 18:09:15 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/ProducerBase.h"
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Provenance/interface/ModuleDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/RunIndex.h"
#include "FWCore/Utilities/interface/LuminosityBlockIndex.h"


// forward declarations

namespace edm {
  namespace stream {
    class EDProducerBase;
    class EDProducerAdaptorBase : public ProducerBase, public EDConsumerBase
    {
      
    public:
      template <typename T> friend class edm::WorkerT;

      EDProducerAdaptorBase();
      virtual ~EDProducerAdaptorBase();
      
      // ---------- const member functions ---------------------
      
      // ---------- static member functions --------------------
      
      // ---------- member functions ---------------------------
      const ModuleDescription moduleDescription() { return moduleDescription_;}
      
      std::string workerType() const { return "WorkerT<EDProducerAdaptorBase>";}
      void
      registerProductsAndCallbacks(EDProducerAdaptorBase const*, ProductRegistry* reg);
    protected:
      template<typename T> void createStreamModules(T iFunc) {
        m_streamModules[0] = iFunc();
      }
      
      void commit(Run& iRun) {
        commit_(iRun);
      }
      void commit(LuminosityBlock& iLumi) {
        commit_(iLumi);
      }
      
    private:
      EDProducerAdaptorBase(const EDProducerAdaptorBase&); // stop default
      
      const EDProducerAdaptorBase& operator=(const EDProducerAdaptorBase&); // stop default
      
      bool doEvent(EventPrincipal& ep, EventSetup const& c,
                           CurrentProcessingContext const* cpcp) ;
      void doBeginJob();
      virtual void doEndJob() = 0;
      
      void doBeginStream(StreamID id);
      void doEndStream(StreamID id);
      void doStreamBeginRun(StreamID id,
                            RunPrincipal& ep,
                            EventSetup const& c,
                            CurrentProcessingContext const* cpcp);
      virtual void setupRun(EDProducerBase*, RunIndex) = 0;
      void doStreamEndRun(StreamID id,
                          RunPrincipal& ep,
                          EventSetup const& c,
                          CurrentProcessingContext const* cpcp);
      virtual void streamEndRunSummary(EDProducerBase*,edm::Run const&, edm::EventSetup const&) = 0;

      void doStreamBeginLuminosityBlock(StreamID id,
                                        LuminosityBlockPrincipal& ep,
                                        EventSetup const& c,
                                        CurrentProcessingContext const* cpcp);
      virtual void setupLuminosityBlock(EDProducerBase*, LuminosityBlockIndex) = 0;
      void doStreamEndLuminosityBlock(StreamID id,
                                      LuminosityBlockPrincipal& ep,
                                      EventSetup const& c,
                                      CurrentProcessingContext const* cpcp);
      virtual void streamEndLuminosityBlockSummary(EDProducerBase*,edm::LuminosityBlock const&, edm::EventSetup const&) = 0;
      
      
      virtual void doBeginRun(RunPrincipal& rp, EventSetup const& c,
                              CurrentProcessingContext const* cpc)=0;
      virtual void doEndRun(RunPrincipal& rp, EventSetup const& c,
                            CurrentProcessingContext const* cpc)=0;
      virtual void doBeginLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                                          CurrentProcessingContext const* cpc)=0;
      virtual void doEndLuminosityBlock(LuminosityBlockPrincipal& lbp, EventSetup const& c,
                                        CurrentProcessingContext const* cpc)=0;
      
      //For now, the following are just dummy implemenations with no ability for users to override
      void doRespondToOpenInputFile(FileBlock const& fb);
      void doRespondToCloseInputFile(FileBlock const& fb);
      void doPreForkReleaseResources();
      void doPostForkReacquireResources(unsigned int iChildIndex, unsigned int iNumberOfChildren);

      // ---------- member data --------------------------------
      void setModuleDescription(ModuleDescription const& md) {
        moduleDescription_ = md;
      }
      ModuleDescription moduleDescription_;
      
      std::vector<EDProducerBase*> m_streamModules;

    };
  }
}

#endif
