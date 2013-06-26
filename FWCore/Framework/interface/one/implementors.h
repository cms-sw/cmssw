#ifndef FWCore_Framework_one_implementors_h
#define FWCore_Framework_one_implementors_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     producerAbilities
// 
/**\file producerAbilities.h "FWCore/Framework/interface/one/implementors"

 Description: Base classes used to implement the interfaces for the edm::one::* module  abilities

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu, 09 May 2013 18:40:17 GMT
// $Id: implementors.h,v 1.1 2013/05/17 14:49:44 chrjones Exp $
//

// system include files
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

// forward declarations

namespace edm {
   namespace one {
      namespace impl {
         
         class SharedResourcesUser {
         public:
            SharedResourcesUser() = default;
            SharedResourcesUser(SharedResourcesUser const&) = delete;
            SharedResourcesUser& operator=(SharedResourcesUser const&) = delete;
            
            virtual ~SharedResourcesUser() = default;
            
         protected:
            static const std::string kUnknownResource;
            
            void usesResource(std::string const& iName = kUnknownResource);
         };
         
         template <typename T>
         class RunWatcher : public virtual T {
         public:
            RunWatcher() = default;
            RunWatcher(RunWatcher const&) = delete;
            RunWatcher& operator=(RunWatcher const&) = delete;
            
         private:
            void doBeginRun_(Run const& rp, EventSetup const& c) override final;
            void doEndRun_(Run const& rp, EventSetup const& c) override final;

            
            virtual void beginRun(edm::Run const&, edm::EventSetup const&) = 0;
            virtual void endRun(edm::Run const&, edm::EventSetup const&) = 0;
         };
         
         template <typename T>
         class LuminosityBlockWatcher : public virtual T {
         public:
            LuminosityBlockWatcher() = default;
            LuminosityBlockWatcher(LuminosityBlockWatcher const&) = delete;
            LuminosityBlockWatcher& operator=(LuminosityBlockWatcher const&) = delete;
            
         private:
            void doBeginLuminosityBlock_(LuminosityBlock const& rp, EventSetup const& c) override final;
            void doEndLuminosityBlock_(LuminosityBlock const& rp, EventSetup const& c) override final;

            virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) = 0;
            virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) = 0;
         };

         template <typename T>
         class BeginRunProducer : public virtual T {
         public:
            BeginRunProducer() = default;
            BeginRunProducer( BeginRunProducer const&) = delete;
            BeginRunProducer& operator=(BeginRunProducer const&) = delete;
            
         private:
            void doBeginRunProduce_(Run& rp, EventSetup const& c) override final;

            virtual void beginRunProduce(edm::Run&, edm::EventSetup const&) = 0;
         };

         template <typename T>
         class EndRunProducer : public virtual T {
         public:
            EndRunProducer() = default;
            EndRunProducer( EndRunProducer const&) = delete;
            EndRunProducer& operator=(EndRunProducer const&) = delete;
            
         private:
            
            void doEndRunProduce_(Run& rp, EventSetup const& c) override final;

            virtual void endRunProduce(edm::Run&, edm::EventSetup const&) = 0;
         };

         template <typename T>
         class BeginLuminosityBlockProducer : public virtual T {
         public:
            BeginLuminosityBlockProducer() = default;
            BeginLuminosityBlockProducer( BeginLuminosityBlockProducer const&) = delete;
            BeginLuminosityBlockProducer& operator=(BeginLuminosityBlockProducer const&) = delete;
            
         private:
            void doBeginLuminosityBlockProduce_(LuminosityBlock& lbp, EventSetup const& c) override final;

            virtual void beginLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) = 0;
         };
         
         template <typename T>
         class EndLuminosityBlockProducer : public virtual T {
         public:
            EndLuminosityBlockProducer() = default;
            EndLuminosityBlockProducer( EndLuminosityBlockProducer const&) = delete;
            EndLuminosityBlockProducer& operator=(EndLuminosityBlockProducer const&) = delete;
            
         private:
            void doEndLuminosityBlockProduce_(LuminosityBlock& lbp, EventSetup const& c) override final;

            virtual void endLuminosityBlockProduce(edm::LuminosityBlock&, edm::EventSetup const&) = 0;
         };
      }
   }
}

#endif
