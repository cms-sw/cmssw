#ifndef ServiceRegistry_connect_but_block_self_h
#define ServiceRegistry_connect_but_block_self_h
// -*- C++ -*-
//
// Package:     ServiceRegistry
// Class  :     connect_but_block_self
// 
/**\function connect_but_block_self connect_but_block_self.h FWCore/ServiceRegistry/interface/connect_but_block_self.h

 Description: Connects a functional object to a signal, but guarantees that the functional object will never see a
   signal caused by its own action.

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Sep 22 20:42:32 CEST 2005
// $Id: connect_but_block_self.h,v 1.4 2013/01/20 17:05:07 chrjones Exp $
//

// system include files
#include <memory>
#include "FWCore/Utilities/interface/Signal.h"

// user include files

// forward declarations
namespace edm {
   namespace serviceregistry {
      template<typename Func>
      class BlockingWrapper
      {
      
       public:
        
         BlockingWrapper(Func iFunc): func_(iFunc), numBlocks_(0) {}
         //virtual ~BlockingWrapper();
         
         // ---------- const member functions ---------------------
         template<typename... Args>
         void operator()(Args&&... args) {
            std::shared_ptr<void> guard(static_cast<void*>(0), std::bind(&BlockingWrapper::unblock,this) );
           if( startBlocking() ) { func_(std::forward<Args>(args)...); }
         }

         // ---------- static member functions --------------------
         
         // ---------- member functions ---------------------------
         
       private:
         // ---------- member data --------------------------------
         bool startBlocking() { return 1 == ++numBlocks_; }
         void unblock() { --numBlocks_;}
         Func func_;
         int numBlocks_;
      };
     
      template<class Func, class Signal>
      void 
      connect_but_block_self(Signal& oSignal, const Func& iFunc) {
        oSignal.connect(BlockingWrapper<Func>(iFunc));
      }
   }
}

#endif
