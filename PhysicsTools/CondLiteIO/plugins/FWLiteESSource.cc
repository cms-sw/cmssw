// -*- C++ -*-
//
// Package:     CondLiteIO
// Class  :     FWLiteESSource
// 
/**\class FWLiteESSource FWLiteESSource.h PhysicsTools/CondLiteIO/interface/FWLiteESSource.h

 Description: [one line class summary]

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Thu Jun 17 15:47:35 CDT 2010
//

// system include files
#include <iostream>
#include <memory>
#include "TFile.h"

// user include files
#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/DataProxy.h"
#include "FWCore/Framework/interface/EventSetupRecordIntervalFinder.h"
#include "DataFormats/FWLite/interface/EventSetup.h"
#include "DataFormats/FWLite/interface/Record.h"
#include "FWCore/Framework/interface/HCTypeTag.h"
#include "FWCore/Utilities/interface/TypeIDBase.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/SourceFactory.h"


// forward declarations
namespace  {
   struct TypeID : public edm::TypeIDBase {
      explicit TypeID(const std::type_info& iInfo): edm::TypeIDBase(iInfo) {}
      TypeID() {}
      using TypeIDBase::typeInfo;
   };
   struct FWLiteESGenericHandle {
      FWLiteESGenericHandle(const TypeID& iType):
      m_type(iType),
      m_data(nullptr),
      m_exception(nullptr){}
      
      FWLiteESGenericHandle(const void* iData):
      m_type(),
      m_data(iData),
      m_exception(nullptr) {}

      FWLiteESGenericHandle(cms::Exception* iException):
      m_type(),
      m_data(nullptr),
      m_exception(iException){}
      
      const std::type_info& typeInfo() const {
         return m_type.typeInfo();
      }
      
      TypeID m_type;
      const void* m_data;
      cms::Exception* m_exception;
   };
   
   class FWLiteProxy : public edm::eventsetup::DataProxy {
   public:
      FWLiteProxy(const TypeID& iTypeID, const fwlite::Record* iRecord):
      m_type(iTypeID),
      m_record(iRecord){}
      
      const void* getImpl(const edm::eventsetup::EventSetupRecordImpl&, const edm::eventsetup::DataKey& iKey) override {
         assert(iKey.type() == m_type);
         
         FWLiteESGenericHandle h(m_type);
         m_record->get(h,iKey.name().value());
         
         if(nullptr!=h.m_exception) {
            throw *(h.m_exception);
         }
         return h.m_data;
      }
      
      void invalidateCache() override {
      }
      
   private:
      TypeID m_type;
      const fwlite::Record* m_record;
   };
}

class FWLiteESSource : public edm::eventsetup::DataProxyProvider, public edm::EventSetupRecordIntervalFinder {
   
public:
   FWLiteESSource(edm::ParameterSet const& iPS);
   ~FWLiteESSource() override;
   
   // ---------- const member functions ---------------------
   
   // ---------- static member functions --------------------
   
   // ---------- member functions ---------------------------
   void newInterval(const edm::eventsetup::EventSetupRecordKey& iRecordType,
                            const edm::ValidityInterval& iInterval) override;
   
   
private:
   FWLiteESSource(const FWLiteESSource&) = delete; // stop default
   
   const FWLiteESSource& operator=(const FWLiteESSource&) = delete; // stop default
   
   void registerProxies(const edm::eventsetup::EventSetupRecordKey& iRecordKey ,
                                KeyedProxies& aProxyList) override;
   
   
   void setIntervalFor(const edm::eventsetup::EventSetupRecordKey&,
                               const edm::IOVSyncValue& , 
                               edm::ValidityInterval&) override;
   
   void delaySettingRecords() override;
   
   // ---------- member data --------------------------------
   std::unique_ptr<TFile> m_file;
   fwlite::EventSetup m_es;
   std::map<edm::eventsetup::EventSetupRecordKey, fwlite::RecordID> m_keyToID;
   
};


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
FWLiteESSource::FWLiteESSource(edm::ParameterSet const& iPS):
m_file( TFile::Open(iPS.getParameter<std::string>("fileName").c_str())),
m_es( m_file.get())
{
}

// FWLiteESSource::FWLiteESSource(const FWLiteESSource& rhs)
// {
//    // do actual copying here;
// }

FWLiteESSource::~FWLiteESSource()
{
}

//
// assignment operators
//
// const FWLiteESSource& FWLiteESSource::operator=(const FWLiteESSource& rhs)
// {
//   //An exception safe implementation is
//   FWLiteESSource temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
FWLiteESSource::newInterval(const edm::eventsetup::EventSetupRecordKey& iRecordType,
                            const edm::ValidityInterval& /*iInterval*/)
{
   invalidateProxies(iRecordType);
}

//
// const member functions
//
void 
FWLiteESSource::registerProxies(const edm::eventsetup::EventSetupRecordKey& iRecordKey ,
                                KeyedProxies& aProxyList)
{
   using edm::eventsetup::heterocontainer::HCTypeTag;

   fwlite::RecordID recID =  m_keyToID[iRecordKey];
   const fwlite::Record& rec = m_es.get(recID);
   typedef std::vector<std::pair<std::string,std::string> > TypesAndLabels;
   TypesAndLabels typesAndLabels = rec.typeAndLabelOfAvailableData();

   std::cout <<"Looking for data in record "<<iRecordKey.name()<<std::endl;
   for(TypesAndLabels::const_iterator it = typesAndLabels.begin(), itEnd = typesAndLabels.end();
       it != itEnd;
       ++it) {
      std::cout <<" need type "<<it->first<<std::endl;
      HCTypeTag tt = HCTypeTag::findType(it->first);
      if(tt != HCTypeTag() ) {
         edm::eventsetup::DataKey dk(tt,edm::eventsetup::IdTags(it->second.c_str()));
         aProxyList.push_back(std::make_pair(dk,
                                             std::make_shared<FWLiteProxy>(TypeID(tt.value()),&rec)));
      } else {
         LogDebug("UnknownESType")<<"The type '"<<it->first<<"' is unknown in this job";
         std::cout <<"    *****FAILED*****"<<std::endl;
      }
   }
   
}


void 
FWLiteESSource::setIntervalFor(const edm::eventsetup::EventSetupRecordKey& iKey,
                               const edm::IOVSyncValue& iSync, 
                               edm::ValidityInterval& oIOV)
{
   m_es.syncTo(iSync.eventID(), iSync.time());
   
   const fwlite::Record& rec = m_es.get(m_keyToID[iKey]);
   edm::IOVSyncValue endSync(rec.endSyncValue().eventID(),
                             rec.endSyncValue().time());
   if(rec.endSyncValue().eventID().run()==0 &&
      rec.endSyncValue().time().value() == 0ULL) {
      endSync = edm::IOVSyncValue::endOfTime();
   }
   oIOV = edm::ValidityInterval(edm::IOVSyncValue(rec.startSyncValue().eventID(),
                                                  rec.startSyncValue().time()),
                                endSync);
}

void 
FWLiteESSource::delaySettingRecords()
{
   using edm::eventsetup::heterocontainer::HCTypeTag;
   std::vector<std::string> recordNames = m_es.namesOfAvailableRecords();
   
   for(std::vector<std::string>::const_iterator it = recordNames.begin(),
       itEnd = recordNames.end();
       it != itEnd;
       ++it) {
      HCTypeTag t = HCTypeTag::findType(*it);
      if(t != HCTypeTag() ) {
         edm::eventsetup::EventSetupRecordKey key(t);
         findingRecordWithKey(key);
         usingRecordWithKey(key);
         m_keyToID[key]=m_es.recordID(it->c_str());
      }
   }
}



//
// static member functions
//

DEFINE_FWK_EVENTSETUP_SOURCE(FWLiteESSource);
