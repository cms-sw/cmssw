#ifndef EVENTSETUP_DATAPROXYTEMPLATE_H
#define EVENTSETUP_DATAPROXYTEMPLATE_H
// -*- C++ -*-
//
// Package:     CoreFramework
// Class  :     DataProxyTemplate
// 
/**\class DataProxyTemplate DataProxyTemplate.h Core/CoreFramework/interface/DataProxyTemplate.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Author:      Chris Jones
// Created:     Thu Mar 31 12:45:32 EST 2005
// $Id: DataProxyTemplate.h,v 1.1 2005/04/02 14:18:01 chrjones Exp $
//

// system include files

// user include files
#include "FWCore/CoreFramework/interface/DataProxy.h"
#include "FWCore/CoreFramework/interface/MakeDataException.h"

// forward declarations

namespace edm {
   namespace eventsetup {
template<class RecordT, class DataT>
class DataProxyTemplate : public DataProxy
{

   public:
      typedef DataT value_type;
      typedef RecordT record_type;
   
      DataProxyTemplate() : cache_(0) {}
      //virtual ~DataProxyTemplate();

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      virtual const DataT* get( const RecordT& iRecord,
                                const DataKey& iKey ) const {
         if( !cacheIsValid() ) {
            cache_ = const_cast<DataProxyTemplate<RecordT, DataT>*>(this)->make(iRecord, iKey );
            const_cast<DataProxyTemplate<RecordT, DataT>*>(this)->setCacheIsValid();
         }
         if( 0 == cache_ ) {
            throwMakeException( iRecord, iKey );
         }
         return cache_;
      }
      
   protected:
      virtual const DataT* make( const RecordT&, const DataKey&) = 0;
      
      virtual void throwMakeException( const RecordT& iRecord,
                                       const DataKey& iKey ) const {
         throw MakeDataException<record_type, value_type>(iKey);
      }

   private:
      DataProxyTemplate( const DataProxyTemplate& ); // stop default

      const DataProxyTemplate& operator=( const DataProxyTemplate& ); // stop default

      // ---------- member data --------------------------------
      mutable const DataT* cache_;
};

   }
}
#endif /* EVENTSETUP_DATAPROXYTEMPLATE_H */
