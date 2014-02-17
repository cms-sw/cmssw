
#ifndef Framework_eventsetup_dependsOn_h
#define Framework_eventsetup_dependsOn_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     eventsetup_dependsOn
// 
/**\class eventsetup_dependsOn eventsetup_dependsOn.h FWCore/Framework/interface/eventsetup_dependsOn.h

 Description: function used to set what methods to call when a dependent Record changes

 Usage:
    The function dependsOn takes a variable number of pointers to member methods as arguments.  These methods are then called 
  for the ESProducer whenever the dependent Record (determined by looking at the methods lone argument) has changed since the
  last time the ESProducer's produce method was called.
\code
       MyProducer : public ESProducer { ... };

       MyProducer::MyProducer(...) {
          setWhatProduced(this, eventsetup::dependsOn(&MyProducer::callWhenChanges));
          ...
      }
\endcode
*/
/*
 Implementation details:
 
 The dependsOn function does not have enough information to convert the list of member function pointers directly into the
 appropriate Decorator class since it is missing the Record type of the Decorator (all it knows are the Record types the 
 Decorator depends on).  Therefore we must defer the creation of the Decorator until that Record type is known (which
 happens within the body of the ESProducer::setWhatProduced method).  To allow the deferred construction, 
 the dependsOn method returns a compile time linked list created via the TwoHolder class (if there is only one 
 node you get a OneHolder).  The dependsOn method always makes sure that the second type of the TwoHolder is the member
 function pointer which is needed for the later construction stage.
 
 Within the body of ESProducer::setWhatProduced, the proper Decorator is created by calling 'createDecoratorFrom' which is given
 a pointer to the Producer, a dummy pointer to the proper Record and the linked list of member function pointers.  The
 'createDecoratorFrom' uses the makeCaller and createDependsOnCaller functions to recursively create the proper DependsOnCaller 
 functor which is then used by the ESPreFunctorDecorator to do the work.  We use  HolderToCaller template class merely to define 
 the return type of the 'createDecoratorFrom' and 'makeCaller' functions.
 
 */

//
// Original Author:  Chris Jones
//         Created:  Thu Jun 23 14:06:56 EDT 2005
// $Id: eventsetup_dependsOn.h,v 1.10 2010/09/01 18:24:25 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/Framework/interface/ESPreFunctorDecorator.h"

// forward declarations


namespace edm {
   namespace eventsetup {
      
      //Simple functor that checks to see if a Record has changed since the last time it was called
      // and if so, calls the appropriate member method.  Multiple callers can be chained together using the
      // TCallerChain template argument.
     template<class T, class TRecord, class TDependsOnRecord, class TCallerChain >
	struct DependsOnCaller
	{
	DependsOnCaller(T* iCallee, void(T::* iMethod)(const TDependsOnRecord&) , const TCallerChain& iChain) : 
	  callee_(iCallee), method_(iMethod), chain_(iChain),cacheID_(0){} 
      
	    void operator()(const TRecord& iRecord) {
	    const TDependsOnRecord& record = iRecord.template getRecord<TDependsOnRecord>();
	    if(record.cacheIdentifier() != cacheID_) {
	      (callee_->*method_)(record);
	      cacheID_=record.cacheIdentifier();
	    }
	    //call next 'functor' in our chain
	    chain_(iRecord);
	  }
	private:
	  T* callee_;
	  void (T::*method_)(const TDependsOnRecord&);
	  TCallerChain chain_;
	  unsigned long long cacheID_;
	};

      //helper function to help create a DependsOnCaller
      template<class T, class TRecord, class TDependsOnRecord, class TCallerChain >
         DependsOnCaller<T,TRecord, TDependsOnRecord, TCallerChain>
         createDependsOnCaller(T* iCallee, const TRecord*, void(T::*iMethod)(const TDependsOnRecord&), const TCallerChain& iChain) 
      {
            return DependsOnCaller<T,TRecord, TDependsOnRecord, TCallerChain>(iCallee, iMethod, iChain);
      }
      
      //A 'do nothing' functor that is used to terminate our chain of functors
      template<class TRecord>
         struct DependsOnDoNothingCaller { void operator()(const TRecord&) {} };
      
      //put implementation details used to get the dependsOn method to work into their own namespace
      namespace depends_on {
         //class to hold onto one member method pointer
         template <class T, class TDependsOnRecord>
         struct OneHolder {
            typedef T Prod_t;
            typedef TDependsOnRecord DependsOnRecord_t;
            
            OneHolder(void (T::*iHoldee)(const TDependsOnRecord&)) : holdee_(iHoldee) {}
            void (T::*holdee_)(const TDependsOnRecord&);
            
         };
         
         //class to create a linked list of member method pointers
         template <class T, class U>
            struct TwoHolder {
               typedef T T1_t;
               typedef U T2_t;
               TwoHolder(const T& i1, const U& i2) : h1_(i1), h2_(i2) {}
               T h1_;
               U h2_;
            };

         //allows one to create the linked list by applying operator & to member method pointers
         template< class T, class U>
            TwoHolder<T,U> operator&(const T& iT, const U& iU) {
               return TwoHolder<T,U>(iT, iU);
            }
         
         //HolderToCaller is used to state how a OneHolder or TwoHolder is converted into the appropriate 
         // DependsOnCaller.  This class is needed to define the return value of the makeCaller function
         template< class TRecord, class THolder>
            struct HolderToCaller {
            };
         template< class TRecord, class T, class TDependsOnRecord >
            struct HolderToCaller<TRecord, OneHolder<T, TDependsOnRecord> > {
               typedef DependsOnCaller<T,TRecord, TDependsOnRecord, DependsOnDoNothingCaller<TRecord> > Caller_t;
            };
         template< class TRecord, class T, class T1, class T2>
            struct HolderToCaller< TRecord, TwoHolder<T1, void (T::*)(const T2&) > > {
               typedef DependsOnCaller<T, TRecord, T2 , typename HolderToCaller<TRecord,T1>::Caller_t > Caller_t;
            };

         //helper function to convert a OneHolder or TwoHolder into a DependsOnCaller.
         template<class T, class TDependsOnRecord, class TRecord>
            DependsOnCaller<T,TRecord, TDependsOnRecord, DependsOnDoNothingCaller<TRecord> >
            makeCaller(T*iT, const TRecord* iRec, const OneHolder<T, TDependsOnRecord>& iHolder) {
               return createDependsOnCaller(iT, iRec, iHolder.holdee_, DependsOnDoNothingCaller<TRecord>());
            }
         
         template<class T, class T1, class T2, class TRecord>
            DependsOnCaller<T,TRecord, T2, typename HolderToCaller<TRecord, T1>::Caller_t >
            makeCaller(T*iT, const TRecord* iRec, const TwoHolder<T1, void (T::*)(const T2&)>& iHolder) {
               return createDependsOnCaller(iT, iRec, iHolder.h2_, makeCaller(iT, iRec, iHolder.h1_));
            }
      }
      
      //DecoratorFromArg is used to declare the return type of 'createDecoratorFrom' based on the arguments to the function.
      template< typename T, typename TRecord, typename TArg>
         struct DecoratorFromArg { typedef TArg Decorator_t; };
      
      template< typename T, typename TRecord, typename TDependsOnRecord>
         struct DecoratorFromArg<T,TRecord, depends_on::OneHolder<T,TDependsOnRecord> > { 
            typedef ESPreFunctorDecorator<TRecord,DependsOnCaller<T,TRecord, TDependsOnRecord, DependsOnDoNothingCaller<TRecord> > > Decorator_t; 
         };
      
      
      template< typename T, typename TRecord, typename TDependsOnRecord >
         inline ESPreFunctorDecorator<TRecord,DependsOnCaller<T,TRecord, TDependsOnRecord, DependsOnDoNothingCaller<TRecord> > > 
         createDecoratorFrom(T* iT, const TRecord*iRec, const depends_on::OneHolder<T,TDependsOnRecord>& iHolder) {
            DependsOnDoNothingCaller<TRecord> tCaller;
            ESPreFunctorDecorator<TRecord,DependsOnCaller<T,TRecord, TDependsOnRecord, DependsOnDoNothingCaller<TRecord> > >
               temp(createDependsOnCaller(iT, iRec, iHolder.holdee_, tCaller));
            return temp;
         }
      
      template< typename T, typename TRecord, typename T1, typename T2>
         struct DecoratorFromArg<T,TRecord, depends_on::TwoHolder<T1,T2> > { 
            typedef ESPreFunctorDecorator<TRecord,typename depends_on::HolderToCaller<TRecord, depends_on::TwoHolder<T1, T2> >::Caller_t >
            Decorator_t; 
         };
      template< typename T, typename TRecord, typename T1, typename T2>
         inline ESPreFunctorDecorator<TRecord,typename depends_on::HolderToCaller<TRecord, depends_on::TwoHolder<T1, T2> >::Caller_t >
         createDecoratorFrom(T* iT, const TRecord*iRec, const depends_on::TwoHolder<T1,T2>& iHolder) {
            return ESPreFunctorDecorator<TRecord, typename depends_on::HolderToCaller<TRecord,depends_on::TwoHolder< T1, T2> >::Caller_t >
            (createDependsOnCaller(iT, iRec, iHolder.h2_, makeCaller(iT, iRec, iHolder.h1_)));
         }
      
      
      //The actual dependsOn functions which users call
      template <typename T, typename TDependsOnRecord>
         depends_on::OneHolder<T,TDependsOnRecord> 
         dependsOn(void(T::*iT)(const TDependsOnRecord&)) { return iT ; }
      
      template< typename T, typename T1, typename T2>
         depends_on::TwoHolder<depends_on::OneHolder<T,T1>, T2> 
         dependsOn(void (T::* iT1)(const T1&), T2 iT2) { return depends_on::OneHolder<T, T1>(iT1) & iT2; }
      
      template< typename T, typename T1, typename T2, typename T3>
         depends_on::TwoHolder< depends_on::TwoHolder<depends_on::OneHolder<T,T1>, T2>, T3>
         dependsOn(void(T::* iT1)(const T1&), T2 iT2, T3 iT3) { return depends_on::OneHolder<T,T1>(iT1) & iT2 & iT3; }
      

   }
}

#endif
