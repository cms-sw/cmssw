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
// $Id$
//

// system include files
#include "boost/shared_ptr.hpp"
#include "boost/signal.hpp"
#include "boost/bind.hpp"

// user include files

// forward declarations
namespace edm {
   namespace serviceregistry {
      template<class Func, class T1=void*, class T2=void*, class T3=void*>
      class BlockingWrapper
      {
      
       public:
         BlockingWrapper(Func iFunc): func_(iFunc), numBlocks_(0) {}
         //virtual ~BlockingWrapper();
         
         // ---------- const member functions ---------------------
         void operator()() {
            boost::shared_ptr<void> guard(static_cast<void*>(0), boost::bind(&BlockingWrapper::unblock,this) );
            if( startBlocking() ) { func_(); }
         }

         void operator()(T1 iT) {
            boost::shared_ptr<void> guard(static_cast<void*>(0), boost::bind(&BlockingWrapper::unblock,this) );
            if( startBlocking() ) { func_(iT); }
         }

         void operator()(T1 iT1, T2 iT2) {
            boost::shared_ptr<void> guard(static_cast<void*>(0), boost::bind(&BlockingWrapper::unblock,this) );
            if( startBlocking() ) { func_(iT,iT2); }
         }

         void operator()(T1 iT1, T2 iT2, T3 iT3) {
            boost::shared_ptr<void> guard(static_cast<void*>(0), boost::bind(&BlockingWrapper::unblock,this) );
            if( startBlocking() ) { func_(iT,iT2,iT3); }
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
      
      template<class T>
         typename T::value_type
         deref(T& iT){ return *iT;}
      
      template<class T>
         BlockingWrapper<T> make_blockingwrapper(T iT,
                                                 const boost::function0<void,std::allocator<void> >*){
            return BlockingWrapper<T>(iT);
         }
      template<class T, class TArg>
         BlockingWrapper<T,TArg> make_blockingwrapper(T iT,
                                                      const boost::function1<void,TArg,std::allocator<void> >*){
            return BlockingWrapper<T,TArg>(iT);
         }
      
      
      template<class Func, class Signal>
      void 
      connect_but_block_self(Signal& oSignal, const Func& iFunc) {
         using boost::signals::connection;
         boost::shared_ptr<boost::shared_ptr<connection> > holder(new boost::shared_ptr<connection>());
         *holder = boost::shared_ptr<connection>(
                    new connection(oSignal.connect(
                                                   make_blockingwrapper(iFunc,
                                                                        static_cast<const typename Signal::slot_function_type*>(0)))));
      }
   }
}

#endif
