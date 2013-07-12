#ifndef Framework_ESPreFunctorDecorator_h
#define Framework_ESPreFunctorDecorator_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESPreFunctorDecorator
// 
/**\class ESPreFunctorDecorator ESPreFunctorDecorator.h FWCore/Framework/interface/ESPreFunctorDecorator.h

 Description: A Decorator that works as a adapter to call a Functor before each call to the decorated method

 Usage:
    This Decorator can be used to create a decorator used in the ESProducer::setWhatProduced method.  This Decorator
will adapt a Functor (a class that implements 'operator()') st that it is called before every call made to
the ESProducer's 'produce' method.

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Jun 22 11:40:41 EDT 2005
// $Id: ESPreFunctorDecorator.h,v 1.5 2005/09/01 05:39:14 wmtan Exp $
//

// system include files

// user include files

// forward declarations

namespace edm {
   namespace eventsetup {

template<class TRecord, class TFunctor >
class ESPreFunctorDecorator
{

   public:
      ESPreFunctorDecorator(const TFunctor& iCaller) :
         caller_(iCaller) {}
      //virtual ~ESPreFunctorDecorator();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      void pre(const TRecord& iRecord) {
         caller_(iRecord);
      }
   
      void post(const TRecord&) {
      }
   
   private:
      //ESPreFunctorDecorator(const ESPreFunctorDecorator&); // stop default

      const ESPreFunctorDecorator& operator=(const ESPreFunctorDecorator&); // stop default

      // ---------- member data --------------------------------
      TFunctor caller_;
};

   }
}

#endif
