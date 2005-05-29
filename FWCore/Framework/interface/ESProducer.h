#ifndef EVENTSETUPPRODUCER_ESPRODUCER_H
#define EVENTSETUPPRODUCER_ESPRODUCER_H
// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     ESProducer
// 
/**\class ESProducer ESProducer.h Core/CoreFramework/interface/ESProducer.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Thu Apr  7 17:08:14 CDT 2005
// $Id: .h,v 1.1 2005/04/18 20:16:16 chrjones Exp $
//

// system include files
#include <memory>

// user include files
#include "FWCore/CoreFramework/interface/ProxyFactoryProducer.h"
#include "FWCore/CoreFramework/interface/ProxyArgumentFactoryTemplate.h"

#include "FWCore/CoreFramework/interface/CallbackProxy.h"
#include "FWCore/CoreFramework/interface/Callback.h"
#include "FWCore/CoreFramework/interface/produce_helpers.h"
#include "FWCore/CoreFramework/interface/ESProducts.h"

// forward declarations
namespace edm {
   namespace eventsetup {      
class ESProducer : public ProxyFactoryProducer
{

   public:
      ESProducer();
      virtual ~ESProducer();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
   protected:      
         template<typename T >
         void setWhatProduced( T* iThis) {
            using namespace boost;
            //BOOST_STATIC_ASSERT( (typename boost::is_base_and_derived<ED, T>::type) );
            setWhatProduced( iThis , &T::produce );
         }
      
      template<typename T, typename TReturn, typename TRecord>
         void setWhatProduced( T* iThis, TReturn (T ::* iMethod)(const TRecord& ) ) {
            using namespace boost;
            boost::shared_ptr<Callback<T,TReturn,TRecord> > callback( new
                                                             Callback<T,TReturn,TRecord>( iThis, iMethod) );
            registerProducts( callback,
                              static_cast<const typename produce::product_traits<TReturn>::type *>(0),
                              static_cast<const TRecord*>(0) );
            //BOOST_STATIC_ASSERT( (boost::is_base_and_derived<ED, T>::type) );
         }

      /*
      template<typename T, typename TReturn, typename TArg>
         void setWhatProduced( T* iThis, TReturn (T ::* iMethod)(const TArg&) ) {
            using namespace boost;
            registerProducts( iThis, static_cast<const typename produce::product_traits<TReturn>::type *>(0) );
            registerGet(iThis, static_cast<const TArg*>(0) );
            //BOOST_STATIC_ASSERT( (boost::is_base_and_derived<ED, T>::type) );
         }
      */
   private:
      ESProducer( const ESProducer& ); // stop default

      ESProducer const& operator=( const ESProducer& ); // stop default

      /*
      template<typename T, typename TProduct>
         void registerGet(T* i, const TProduct* iProd) {
            using namespace produce;
            std::cout <<"registered 'get' for product type "
            << test::name( iProd ) <<
            std::endl;
         };
      */
      template<typename CallbackT, typename TList, typename TRecord>
         void registerProducts(boost::shared_ptr<CallbackT> iCallback, const TList*, const TRecord* iRecord) {
            registerProduct(iCallback, static_cast<const typename TList::tail_type*>(0), iRecord );
            registerProducts(iCallback, static_cast<const typename TList::head_type*>(0), iRecord );
         };
      template<typename T, typename TRecord>
         void registerProducts(boost::shared_ptr<T>, const produce::Null*, const TRecord*) {
            //do nothing
         };
      
      
      template<typename T, typename TProduct, typename TRecord>
         void registerProduct(boost::shared_ptr<T> iCallback, const TProduct* iProd, const TRecord*) {
            registerFactory( new ProxyArgumentFactoryTemplate<
                             CallbackProxy<T, TRecord, TProduct>, boost::shared_ptr<T> >( iCallback ) );
         };
      
      // ---------- member data --------------------------------
      // NOTE: the factories share ownership of the callback
      //std::vector<boost::shared_ptr<CallbackBase> > callbacks_;
      
};
   }
}
#endif /* EVENTSETUPPRODUCER_ESPRODUCER_H */
