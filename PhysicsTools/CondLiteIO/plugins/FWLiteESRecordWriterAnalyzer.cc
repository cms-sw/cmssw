// -*- C++ -*-
//
// Package:    FWLiteESRecordWriterAnalyzer
// Class:      FWLiteESRecordWriterAnalyzer
// 
/**\class FWLiteESRecordWriterAnalyzer FWLiteESRecordWriterAnalyzer.cc PhysicsTools/FWLiteESRecordWriterAnalyzer/src/FWLiteESRecordWriterAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Chris Jones
//         Created:  Fri Jun 18 14:23:07 CDT 2010
//
//


// system include files
#include <memory>
#include "TFile.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/CondLiteIO/interface/RecordWriter.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Framework/interface/HCTypeTag.h"

//
// class declaration
//

namespace  {
   
   struct DataInfo {
      DataInfo( const edm::eventsetup::heterocontainer::HCTypeTag& iTag,
               const std::string& iLabel) :
      m_tag(iTag), m_label(iLabel) {}
      edm::eventsetup::heterocontainer::HCTypeTag m_tag;
      std::string m_label;
   };
}

namespace fwliteeswriter {
   struct DummyType {
      const edm::eventsetup::heterocontainer::HCTypeTag* m_tag;
      mutable const void* m_data;
   };
   struct Handle {
      Handle(const DataInfo* iInfo): m_data(nullptr), m_info(iInfo) {}
      const void* m_data;
      const DataInfo* m_info;
      const edm::eventsetup::ComponentDescription* m_desc;
   };
}


namespace edm {
   namespace eventsetup {
      
      template <> 
      void EventSetupRecordImpl::getImplementation<fwliteeswriter::DummyType>(fwliteeswriter::DummyType const *& iData ,
                                                                              const char* iName,
                                                                              const ComponentDescription*& iDesc,
                                                                              bool iTransientAccessOnly,
                                                                              std::shared_ptr<ESHandleExceptionFactory>& whyFailedFactory) const {
         DataKey dataKey(*(iData->m_tag),
                         iName,
                         DataKey::kDoNotCopyMemory);
         
         const void* pValue = this->getFromProxy(dataKey,iDesc,iTransientAccessOnly);
         if(nullptr==pValue) {
	   throw cms::Exception("NoProxyException")<<"No data of type \""<<iData->m_tag->name()<<"\" with label \""<<
	     iName<<"\" in record \""<<this->key().name()<<"\"";
         }
         iData->m_data = pValue;
      }
      
      template<>
      bool EventSetupRecord::get<fwliteeswriter::Handle>(const std::string& iName, fwliteeswriter::Handle& iHolder) const {
         fwliteeswriter::DummyType t;
         t.m_tag = &(iHolder.m_info->m_tag);
         const fwliteeswriter::DummyType* value = &t;
         const ComponentDescription* desc = nullptr;
         std::shared_ptr<ESHandleExceptionFactory> dummy;
         impl_->getImplementation(value, iName.c_str(),desc,true, dummy);
         iHolder.m_data = t.m_data;
         iHolder.m_desc = desc;
         return true;
      }
      
   }
}


namespace  {

   class RecordHandler {
   public:
      RecordHandler(const edm::eventsetup::EventSetupRecordKey& iRec,
                    TFile* iFile,
                    std::vector<DataInfo>& ioInfo):
      m_key(iRec),
      m_record(),
      m_writer(m_key.name(),iFile),
      m_cacheID(0) {
         m_dataInfos.swap(ioInfo);
      }
      
      void update(const edm::EventSetup& iSetup) {
         if(not m_record) {
            m_record = iSetup.find(m_key);
            assert(m_record);
         }
         if(m_cacheID != m_record->cacheIdentifier()) {
            m_cacheID = m_record->cacheIdentifier();
         
            for(std::vector<DataInfo>::const_iterator it = m_dataInfos.begin(),
                itEnd = m_dataInfos.end();
                it != itEnd;
                ++it) {
               fwliteeswriter::Handle h(&(*it));
               m_record->get(it->m_label,h);
               m_writer.update(h.m_data,(it->m_tag.value()),it->m_label.c_str());
            }
            edm::ValidityInterval const& iov= m_record->validityInterval();
            m_writer.fill(edm::ESRecordAuxiliary(iov.first().eventID(),
                                                 iov.first().time()) );
         }
      }
   private:
      edm::eventsetup::EventSetupRecordKey m_key;
     std::optional<edm::eventsetup::EventSetupRecordGeneric> m_record;
      fwlite::RecordWriter m_writer;
      unsigned long long m_cacheID;
      std::vector<DataInfo> m_dataInfos;
   };
}

                        
class FWLiteESRecordWriterAnalyzer : public edm::EDAnalyzer {
   public:
      explicit FWLiteESRecordWriterAnalyzer(const edm::ParameterSet&);
      ~FWLiteESRecordWriterAnalyzer() override;


   private:
      void beginJob() override ;
      void analyze(const edm::Event&, const edm::EventSetup&) override;
      void endJob() override ;
      void beginRun(edm::Run const&, edm::EventSetup const&) override;
      void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
   
      void update(const edm::EventSetup&);

      // ----------member data ---------------------------
   std::vector<std::shared_ptr<RecordHandler> > m_handlers;
   
   std::map<std::string, std::vector<std::pair<std::string,std::string> > > m_recordToDataNames;
   TFile* m_file;
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
FWLiteESRecordWriterAnalyzer::FWLiteESRecordWriterAnalyzer(const edm::ParameterSet& iConfig)
{
   std::vector<std::string> names = iConfig.getParameterNamesForType<std::vector<edm::ParameterSet> >(false);
   if (names.empty()) {
      throw edm::Exception(edm::errors::Configuration)<<"No VPSets were given in configuration";
   }
   for (std::vector<std::string>::const_iterator it = names.begin(), itEnd=names.end(); it != itEnd; ++it) {
      const std::vector<edm::ParameterSet>& ps = iConfig.getUntrackedParameter<std::vector<edm::ParameterSet> >(*it);
      std::vector<std::pair<std::string,std::string> >& data = m_recordToDataNames[*it];
      for(std::vector<edm::ParameterSet>::const_iterator itPS = ps.begin(),itPSEnd = ps.end(); 
          itPS != itPSEnd;
          ++itPS){ 
         std::string type = itPS->getUntrackedParameter<std::string>("type");
         std::string label = itPS->getUntrackedParameter<std::string>("label",std::string());
         data.push_back(std::make_pair(type,label) );
      }
   }
   
   m_file = TFile::Open(iConfig.getUntrackedParameter<std::string>("fileName").c_str(),"NEW");
}


FWLiteESRecordWriterAnalyzer::~FWLiteESRecordWriterAnalyzer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
   m_file->Close();
   delete m_file;
}


//
// member functions
//
void
FWLiteESRecordWriterAnalyzer::update(const edm::EventSetup& iSetup)
{
   using edm::eventsetup::heterocontainer::HCTypeTag;
   if(m_handlers.empty()) {
      //now we have access to the EventSetup so we can setup our data structure
      for(std::map<std::string, std::vector<std::pair<std::string,std::string> > >::iterator it=m_recordToDataNames.begin(),
          itEnd = m_recordToDataNames.end();
          it != itEnd;
          ++it) {
         HCTypeTag tt = HCTypeTag::findType(it->first);
         if(tt == HCTypeTag()) {
            throw cms::Exception("UnknownESRecordType")<<"The name '"<<it->first<<"' is not associated with a known EventSetupRecord.\n"
            "Please check spelling or load a module known to link with the package which declares that Record.";
         }
         edm::eventsetup::EventSetupRecordKey rKey(tt);
         
         auto rec = iSetup.find(tt);
         if(not rec) {
            throw cms::Exception("UnknownESRecordType")<<"The name '"<<it->first<<"' is not associated with a type which is not an EventSetupRecord.\n"
            "Please check your spelling.";
         }
         
         //now figure out what data
         std::vector<std::pair<std::string,std::string> >& data = it->second;
         if(data.empty()) {
            //get everything from the record
            std::vector<edm::eventsetup::DataKey> keys;
            rec->fillRegisteredDataKeys(keys);
            for(std::vector<edm::eventsetup::DataKey>::iterator itKey = keys.begin(), itKeyEnd = keys.end();
                itKey != itKeyEnd;
                ++itKey) {
               data.push_back(std::make_pair(std::string(itKey->type().name()),
                                             std::string(itKey->name().value())));
            }
         }
         
         std::vector<DataInfo> dataInfos;
         for (std::vector<std::pair<std::string,std::string> >::iterator itData = data.begin(), itDataEnd = data.end(); 
              itData != itDataEnd;              
              ++itData) {
            HCTypeTag tt = HCTypeTag::findType(itData->first);
            if(tt == HCTypeTag()) {
               throw cms::Exception("UnknownESDataType")<<"The name '"<<itData->first<<"' is not associated with a known type held in the "<<it->first<<" Record.\n"
               "Please check spelling or load a module known to link with the package which declares that type.";
            }
            if(!bool(edm::TypeWithDict( tt.value() ))) {
               throw cms::Exception("NoDictionary")<<"The type '"<<itData->first<<"' can not be retrieved from the Record "<<it->first<<" and stored \n"
               "because no dictionary exists for the type.";
            }
            dataInfos.push_back(DataInfo(tt,itData->second));
         }
         m_handlers.push_back( std::make_shared<RecordHandler>(rKey,m_file,dataInfos) );
      }
   }
   
   for(std::vector<std::shared_ptr<RecordHandler> >::iterator it = m_handlers.begin(),itEnd = m_handlers.end();
       it != itEnd;
       ++it) {
      (*it)->update(iSetup);
   }
}


// ------------ method called to for each event  ------------
void
FWLiteESRecordWriterAnalyzer::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup)
{
   update(iSetup);
}


// ------------ method called once each job just before starting event loop  ------------
void 
FWLiteESRecordWriterAnalyzer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
FWLiteESRecordWriterAnalyzer::endJob() {
   m_file->Write();
}

void 
FWLiteESRecordWriterAnalyzer::beginRun(edm::Run const&, edm::EventSetup const& iSetup){
   update(iSetup);
}
void 
FWLiteESRecordWriterAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const& iSetup){
   update(iSetup);
}


//define this as a plug-in
DEFINE_FWK_MODULE(FWLiteESRecordWriterAnalyzer);
