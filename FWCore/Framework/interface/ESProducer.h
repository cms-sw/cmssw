#ifndef Framework_ESProducer_h
#define Framework_ESProducer_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     ESProducer
//
/**\class ESProducer ESProducer.h FWCore/Framework/interface/ESProducer.h

 Description: An EventSetup algorithmic Provider that encapsulates the algorithm as a member method

 Usage:
    Inheriting from this class is the simplest way to create an algorithm which gets called when a new
  data item is needed for the EventSetup.  This class is designed to call a member method of inheriting
  classes each time the algorithm needs to be run.  (A more flexible system in which the algorithms can be
  set at run-time instead of compile time can be obtained by inheriting from ESProxyFactoryProducer instead.)

    If only one algorithm is being encapsulated then the user needs to
      1) add a method name 'produce' to the class.  The 'produce' takes as its argument a const reference
         to the record that is to hold the data item being produced.  If only one data item is being produced,
         the 'produce' method must return either an 'std::unique_ptr' or 'std::shared_ptr' to the object being
         produced.  (The choice depends on if the EventSetup or the ESProducer is managing the lifetime of
         the object).  If multiple items are being Produced they the 'produce' method must return an
         ESProducts<> object which holds all of the items.
      2) add 'setWhatProduced(this);' to their classes constructor

Example: one algorithm creating only one object
\code
    class FooProd : public edm::ESProducer {
       std::unique_ptr<Foo> produce(const FooRecord&);
       ...
    };
    FooProd::FooProd(const edm::ParameterSet&) {
       setWhatProduced(this);
       ...
    }
\endcode
Example: one algorithm creating two objects
\code
   class FoosProd : public edm::ESProducer {
      edm::ESProducts<std::unique_ptr<Foo1>, std::unique_ptr<Foo2>> produce(const FooRecord&);
      ...
   };
\endcode

  If multiple algorithms are being encapsulated then
      1) like 1 above except the methods can have any names you want
      2) add 'setWhatProduced(this, &<class name>::<method name>);' for each method in the class' constructor
   NOTE: the algorithms can put data into the same record or into different records

Example: two algorithms each creating only one objects
\code
   class FooBarProd : public edm::eventsetup::ESProducer {
      std::unique_ptr<Foo> produceFoo(const FooRecord&);
      std::unique_ptr<Bar> produceBar(const BarRecord&);
      ...
   };
   FooBarProd::FooBarProd(const edm::ParameterSet&) {
      setWhatProduced(this,&FooBarProd::produceFoo);
      setWhatProduced(this,&FooBarProd::produceBar);
      ...
   }
\endcode

*/
//
// Author:      Chris Jones
// Created:     Thu Apr  7 17:08:14 CDT 2005
//

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/ESProxyFactoryProducer.h"
#include "FWCore/Framework/interface/ProxyArgumentFactoryTemplate.h"

#include "FWCore/Framework/interface/CallbackProxy.h"
#include "FWCore/Framework/interface/Callback.h"
#include "FWCore/Framework/interface/produce_helpers.h"
#include "FWCore/Framework/interface/eventsetup_dependsOn.h"
#include "FWCore/Framework/interface/es_Label.h"

// forward declarations
namespace edm {
  namespace eventsetup {

    //used by ESProducer to create the proper Decorator based on the
    //  argument type passed.  The default it to just 'pass through'
    //  the argument as the decorator itself
    template< typename T, typename TRecord, typename TDecorator >
    inline const TDecorator& createDecoratorFrom(T*, const TRecord*, const TDecorator& iDec) {
      return iDec;
    }
  }
  class ESProducer : public ESProxyFactoryProducer
  {

  public:
    ESProducer();
    ~ESProducer() noexcept(false) override;

    // ---------- const member functions ---------------------

    // ---------- static member functions --------------------

    // ---------- member functions ---------------------------
  protected:
    /** \param iThis the 'this' pointer to an inheriting class instance
        The method determines the Record argument and return value of the 'produce'
        method in order to do the registration with the EventSetup
    */
    template<typename T>
    void setWhatProduced(T* iThis, const es::Label& iLabel = es::Label()) {
      setWhatProduced(iThis , &T::produce, iLabel);
    }

    template<typename T>
    void setWhatProduced(T* iThis, const char* iLabel) {
      setWhatProduced(iThis , es::Label(iLabel));
    }
    template<typename T>
    void setWhatProduced(T* iThis, const std::string& iLabel) {
      setWhatProduced(iThis , es::Label(iLabel));
    }

    template<typename T, typename TDecorator >
    void setWhatProduced(T* iThis, const TDecorator& iDec, const es::Label& iLabel = es::Label()) {
      setWhatProduced(iThis , &T::produce, iDec, iLabel);
    }
    /** \param iThis the 'this' pointer to an inheriting class instance
        \param iMethod a member method of then inheriting class
        The method determines the Record argument and return value of the iMethod argument
        method in order to do the registration with the EventSetup
    */
    template<typename T, typename TReturn, typename TRecord>
    void setWhatProduced(T* iThis,
                         TReturn (T ::* iMethod)(const TRecord&),
                         const es::Label& iLabel = es::Label()) {
      setWhatProduced(iThis, iMethod, eventsetup::CallbackSimpleDecorator<TRecord>(),iLabel);
    }
    /** \param iThis the 'this' pointer to an inheriting class instance
        \param iMethod a member method of then inheriting class
        \param iDecorator a class with 'pre'&'post' methods which are placed around the method call
        The method determines the Record argument and return value of the iMethod argument
        method in order to do the registration with the EventSetup
    */
    template<typename T, typename TReturn, typename TRecord, typename TArg>
    void setWhatProduced(T* iThis,
                         TReturn (T ::* iMethod)(const TRecord&),
                         const TArg& iDec,
                         const es::Label& iLabel = es::Label()) {
      auto callback = std::make_shared<eventsetup::Callback<T,
                                                            TReturn,
                                                            TRecord,
                                                            typename eventsetup::DecoratorFromArg<T,TRecord,TArg>::Decorator_t>>(iThis,
                                                                                                                                 iMethod,
                                                                                                                                 createDecoratorFrom(iThis,
                                                                                                                                                     static_cast<const TRecord*>(nullptr),
                                                                                                                                                     iDec));
      registerProducts(callback,
                       static_cast<const typename eventsetup::produce::product_traits<TReturn>::type *>(nullptr),
                       static_cast<const TRecord*>(nullptr),
                       iLabel);
      //static_assert((std::is_base_of<ED, T>::type));
    }

    ESProducer(const ESProducer&) = delete; // stop default
    ESProducer const& operator=(const ESProducer&) = delete; // stop default

  private:

    template<typename CallbackT, typename TList, typename TRecord>
    void registerProducts(std::shared_ptr<CallbackT> iCallback, const TList*, const TRecord* iRecord,
                          const es::Label& iLabel) {
      registerProduct(iCallback, static_cast<const typename TList::tail_type*>(nullptr), iRecord, iLabel);
      registerProducts(iCallback, static_cast<const typename TList::head_type*>(nullptr), iRecord, iLabel);
    }
    template<typename T, typename TRecord>
    void registerProducts(std::shared_ptr<T>, const eventsetup::produce::Null*, const TRecord*,const es::Label&) {
      //do nothing
    }


    template<typename T, typename TProduct, typename TRecord>
    void registerProduct(std::shared_ptr<T> iCallback, const TProduct*, const TRecord*,const es::Label& iLabel) {
      typedef eventsetup::CallbackProxy<T, TRecord, TProduct> ProxyType;
      typedef eventsetup::ProxyArgumentFactoryTemplate<ProxyType, std::shared_ptr<T>> FactoryType;
      registerFactory(std::make_unique<FactoryType>(iCallback), iLabel.default_);
    }

    template<typename T, typename TProduct, typename TRecord, int IIndex>
    void registerProduct(std::shared_ptr<T> iCallback, const es::L<TProduct,IIndex>*, const TRecord*,const es::Label& iLabel) {
      if(iLabel.labels_.size() <= IIndex ||
         iLabel.labels_[IIndex] == es::Label::def()) {
        Exception::throwThis(errors::Configuration,
                             "Unnamed Label\nthe index ",
                             IIndex,
                             " was never assigned a name in the 'setWhatProduced' method");
      }
      typedef eventsetup::CallbackProxy<T, TRecord, es::L<TProduct, IIndex>> ProxyType;
      typedef eventsetup::ProxyArgumentFactoryTemplate<ProxyType, std::shared_ptr<T>> FactoryType;
      registerFactory(std::make_unique<FactoryType>(iCallback), iLabel.labels_[IIndex]);
    }

  };
}
#endif
