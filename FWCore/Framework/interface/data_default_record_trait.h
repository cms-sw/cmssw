#ifndef Framework_data_default_record_trait_h
#define Framework_data_default_record_trait_h
// -*- C++ -*-
//
// Package:     Framework
// Class  :     data_default_record_trait
// 
/**\class data_default_record_trait data_default_record_trait.h FWCore/Framework/interface/data_default_record_trait.h

 Description: trait class that assigns a default EventSetup Record to a particular data type

 Usage:
    Many types of data in a EventSetup are only available in one Record type.  For example, the HCal Alignment
  data only appears in the HCal Alignment Record. For such cases, it is annoying for users to have to specify
  both the Record and the Data type in order to get the data.  In such a case, a specialization of
  data_default_record_trait for that data type can be assigned to allow access 'direct' access to that data
  from the EventSetup

  ESHandle<MyData> pMyData;
  eventSetup.getData(pMyData);

  which is just a short hand for

  ESHandle<MyData> pMyData;
  eventSetup.get<MyDataRecord>.get(pMyData);

    To specify the default record, you must use the macro EVENTSETUP_DATA_DEFAULT_RECORD in the header file for
  the data (or in a header file that users will include).  For Example


    #include "..../MyDataRecord.h"

    class MyData { ...
    };

    EVENTSETUP_DATA_DEFAULT_RECORD(MyData, MyDataRecord);
*/
//
// Author:      Chris Jones
// Created:     Thu Apr  7 07:59:56 CDT 2005
//

namespace edm {

  template <typename T>
  class ESHandle;

  // Special class to denote that the default record should be used.
  struct DefaultRecord {};

   namespace eventsetup {
      template< class T> struct MUST_GET_RECORD_FROM_EVENTSETUP_TO_GET_DATA;
      
      template<class DataT>
         struct data_default_record_trait
      {
         //NOTE: by default, a data item does not have a default record
         typedef MUST_GET_RECORD_FROM_EVENTSETUP_TO_GET_DATA<DataT> type;
      };

    template <typename T>
    struct default_record {
      using data_type = typename T::value_type;
      using RecordT = typename eventsetup::data_default_record_trait<data_type>::type;
    };

    template <typename T>
    using default_record_t = typename default_record<T>::RecordT;
   }
}

#define EVENTSETUP_DATA_DEFAULT_RECORD(_data_, _record_) \
  namespace edm::eventsetup { template<> struct data_default_record_trait<_data_>{ typedef _record_ type; }; }

#endif
