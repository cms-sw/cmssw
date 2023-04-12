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
#include "FWCore/Reflection/interface/TypeWithDict.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EventSetupRecord.h"
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "FWCore/Framework/interface/eventsetuprecord_registration_macro.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESRecordsToProxyIndices.h"
#include "FWCore/Framework/interface/HCTypeTag.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/CondLiteIO/interface/RecordWriter.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/ESGetTokenGeneric.h"

//
// class declaration
//

namespace {

  struct DataInfo {
    DataInfo(const edm::eventsetup::heterocontainer::HCTypeTag& iTag, const std::string& iLabel)
        : m_tag(iTag), m_label(iLabel) {}
    edm::eventsetup::heterocontainer::HCTypeTag m_tag;
    edm::ESGetTokenGeneric m_runToken;
    edm::ESGetTokenGeneric m_lumiToken;
    std::string m_label;
  };
}  // namespace

namespace fwliteeswriter {
  struct DummyType {
    const edm::eventsetup::heterocontainer::HCTypeTag* m_tag;
    mutable const void* m_data;
  };

  class FWLWEventSetupRecord : public edm::eventsetup::EventSetupRecordImplementation<FWLWEventSetupRecord> {
  public:
    FWLWEventSetupRecord& operator=(const edm::eventsetup::EventSetupRecordGeneric& iOther) {
      edm::eventsetup::EventSetupRecord::operator=(iOther);
      return *this;
    }
  };

}  // namespace fwliteeswriter

namespace edm {

  template <>
  class ESHandle<fwliteeswriter::DummyType> : public ESHandleBase {
  public:
    //typedef T value_type;

    ESHandle() = default;
    ESHandle(void const* iData) : ESHandleBase(iData, nullptr) {}
    ESHandle(void const* iData, edm::eventsetup::ComponentDescription const* desc) : ESHandleBase(iData, desc) {}
    ESHandle(std::shared_ptr<ESHandleExceptionFactory>&& iWhyFailed) : ESHandleBase(std::move(iWhyFailed)) {}

    // ---------- const member functions ---------------------
    void const* product() const { return productStorage(); }
    // ---------- static member functions --------------------
    static constexpr bool transientAccessOnly = false;

    // ---------- member functions ---------------------------
  };

  template <>
  class ESGetToken<fwliteeswriter::DummyType, fwliteeswriter::FWLWEventSetupRecord> {
    friend class eventsetup::EventSetupRecord;

  public:
    explicit constexpr ESGetToken(DataInfo const& info, bool isRun) noexcept : m_info{info}, m_isRun{isRun} {}

    constexpr unsigned int transitionID() const noexcept {
      return m_isRun ? m_info.m_runToken.transitionID() : m_info.m_lumiToken.transitionID();
    }
    constexpr bool isInitialized() const noexcept { return transitionID() != std::numeric_limits<unsigned int>::max(); }
    constexpr ESTokenIndex index() const noexcept {
      return m_isRun ? m_info.m_runToken.index() : m_info.m_lumiToken.index();
    }
    constexpr bool hasValidIndex() const noexcept { return index() != invalidIndex(); }
    static constexpr ESTokenIndex invalidIndex() noexcept { return ESTokenIndex{std::numeric_limits<int>::max()}; }

    constexpr DataInfo const& dataInfo() const noexcept { return m_info; }

  private:
    char const* name() const noexcept { return m_info.m_label.c_str(); }
    DataInfo const& m_info;
    bool m_isRun;
  };

  class EventSetupImpl;

  namespace eventsetup {

    template <>
    void EventSetupRecordImpl::getImplementation<fwliteeswriter::DummyType>(
        fwliteeswriter::DummyType const*& iData,
        ESProxyIndex iProxyIndex,
        bool iTransientAccessOnly,
        ComponentDescription const*& oDesc,
        std::shared_ptr<ESHandleExceptionFactory>& whyFailedFactory) const {
      DataKey const* dataKey = nullptr;

      if (iProxyIndex.value() == std::numeric_limits<int>::max()) {
        throw cms::Exception("NoProxyException") << "No data of type \"" << iData->m_tag->name()
                                                 << "\" with unknown label in record \"" << this->key().name() << "\"";
        iData->m_data = nullptr;
        return;
      }
      assert(iProxyIndex.value() > -1 and
             iProxyIndex.value() < static_cast<ESProxyIndex::Value_t>(keysForProxies_.size()));
      void const* pValue = this->getFromProxyAfterPrefetch(iProxyIndex, iTransientAccessOnly, oDesc, dataKey);
      if (nullptr == pValue) {
        throw cms::Exception("NoDataException")
            << "No data of type \"" << iData->m_tag->name() << "\" with label \"" << dataKey->name().value()
            << "\" in record \"" << this->key().name() << "\"";
      }
      iData->m_data = pValue;
    }

    template <>
    edm::ESHandle<fwliteeswriter::DummyType>
    EventSetupRecord::getHandleImpl<edm::ESHandle, fwliteeswriter::DummyType, fwliteeswriter::FWLWEventSetupRecord>(
        ESGetToken<fwliteeswriter::DummyType, fwliteeswriter::FWLWEventSetupRecord> const& iToken) const {
      if UNLIKELY (not iToken.isInitialized()) {
        std::rethrow_exception(makeUninitializedTokenException(this->key(), iToken.dataInfo().m_tag));
      }
      if UNLIKELY (iToken.transitionID() != transitionID()) {
        throwWrongTransitionID();
      }
      using TheHandle = edm::ESHandle<fwliteeswriter::DummyType>;
      assert(getTokenIndices_);
      //need to check token has valid index
      if UNLIKELY (not iToken.hasValidIndex()) {
        return TheHandle{makeESHandleExceptionFactory(
            [key = this->key(), tag = iToken.dataInfo().m_tag, transitionID = iToken.transitionID()] {
              return makeInvalidTokenException(key, tag, transitionID);
            })};
      }

      auto proxyIndex = getTokenIndices_[iToken.index().value()];
      if UNLIKELY (proxyIndex.value() == std::numeric_limits<int>::max()) {
        return TheHandle(makeESHandleExceptionFactory([iToken, key = this->key()] {
          cms::Exception exc("NoProxyException");
          exc << "No data of type \"" << iToken.dataInfo().m_tag.name() << "\" with label \"" << iToken.name()
              << "\" in record \"" << key.name() << "\"";
          return std::make_exception_ptr(exc);
        }));
      }

      fwliteeswriter::DummyType value;
      fwliteeswriter::DummyType const* pValue = &value;
      ComponentDescription const* desc = nullptr;
      std::shared_ptr<ESHandleExceptionFactory> whyFailedFactory;

      impl_->getImplementation(pValue, proxyIndex, false, desc, whyFailedFactory);

      if UNLIKELY (not value.m_data) {
        std::rethrow_exception(whyFailedFactory->make());
      }
      return edm::ESHandle<fwliteeswriter::DummyType>(value.m_data, desc);
    }

  }  // namespace eventsetup
}  // namespace edm

namespace {

  class RecordHandler {
  public:
    RecordHandler(const edm::eventsetup::EventSetupRecordKey& iRec, TFile* iFile, std::vector<DataInfo>& ioInfo)
        : m_key(iRec), m_record(), m_writer(m_key.name(), iFile), m_cacheID(0) {
      m_dataInfos.swap(ioInfo);
    }

    void update(const edm::EventSetup& iSetup, bool iIsRun) {
      if (not m_record) {
        m_record = iSetup.find(m_key);
        assert(m_record);
      }
      if (m_cacheID != m_record->cacheIdentifier()) {
        m_cacheID = m_record->cacheIdentifier();

        for (std::vector<DataInfo>::const_iterator it = m_dataInfos.begin(), itEnd = m_dataInfos.end(); it != itEnd;
             ++it) {
          fwliteeswriter::FWLWEventSetupRecord tempRecord;
          tempRecord = *m_record;
          edm::ESGetToken<fwliteeswriter::DummyType, fwliteeswriter::FWLWEventSetupRecord> token(*it, iIsRun);
          auto h = tempRecord.getHandle(token);
          m_writer.update(h.product(), (it->m_tag.value()), it->m_label.c_str());
        }
        edm::ValidityInterval const& iov = m_record->validityInterval();
        m_writer.fill(edm::ESRecordAuxiliary(iov.first().eventID(), iov.first().time()));
      }
    }

  private:
    edm::eventsetup::EventSetupRecordKey m_key;
    std::optional<edm::eventsetup::EventSetupRecordGeneric> m_record;
    fwlite::RecordWriter m_writer;
    unsigned long long m_cacheID;
    std::vector<DataInfo> m_dataInfos;
  };
}  // namespace

class FWLiteESRecordWriterAnalyzer : public edm::one::EDAnalyzer<edm::one::WatchRuns, edm::one::WatchLuminosityBlocks> {
public:
  explicit FWLiteESRecordWriterAnalyzer(const edm::ParameterSet&);
  ~FWLiteESRecordWriterAnalyzer() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override {}
  void endRun(edm::Run const&, edm::EventSetup const&) override {}

  void update(const edm::EventSetup&, bool isRun);

  void registerLateConsumes(edm::eventsetup::ESRecordsToProxyIndices const&) final;

  // ----------member data ---------------------------
  std::vector<std::shared_ptr<RecordHandler> > m_handlers;

  std::map<std::string, std::vector<std::pair<std::string, std::string> > > m_recordToDataNames;
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
FWLiteESRecordWriterAnalyzer::FWLiteESRecordWriterAnalyzer(const edm::ParameterSet& iConfig) {
  std::vector<std::string> names = iConfig.getParameterNamesForType<std::vector<edm::ParameterSet> >(false);
  if (names.empty()) {
    throw edm::Exception(edm::errors::Configuration) << "No VPSets were given in configuration";
  }
  for (std::vector<std::string>::const_iterator it = names.begin(), itEnd = names.end(); it != itEnd; ++it) {
    const std::vector<edm::ParameterSet>& ps = iConfig.getUntrackedParameter<std::vector<edm::ParameterSet> >(*it);
    std::vector<std::pair<std::string, std::string> >& data = m_recordToDataNames[*it];
    for (std::vector<edm::ParameterSet>::const_iterator itPS = ps.begin(), itPSEnd = ps.end(); itPS != itPSEnd;
         ++itPS) {
      std::string type = itPS->getUntrackedParameter<std::string>("type");
      std::string label = itPS->getUntrackedParameter<std::string>("label", std::string());
      data.push_back(std::make_pair(type, label));
    }
  }

  m_file = TFile::Open(iConfig.getUntrackedParameter<std::string>("fileName").c_str(), "NEW");
}

void FWLiteESRecordWriterAnalyzer::registerLateConsumes(edm::eventsetup::ESRecordsToProxyIndices const& iInfo) {
  using edm::eventsetup::heterocontainer::HCTypeTag;

  for (auto it = m_recordToDataNames.begin(), itEnd = m_recordToDataNames.end(); it != itEnd; ++it) {
    HCTypeTag tt = HCTypeTag::findType(it->first);
    if (tt == HCTypeTag()) {
      throw cms::Exception("UnknownESRecordType")
          << "The name '" << it->first
          << "' is not associated with a known EventSetupRecord.\n"
             "Please check spelling or load a module known to link with the package which declares that Record.";
    }
    edm::eventsetup::EventSetupRecordKey rKey(tt);

    auto recIndex = iInfo.recordIndexFor(rKey);
    if (recIndex == iInfo.missingRecordIndex()) {
      throw cms::Exception("UnknownESRecordType")
          << "The name '" << it->first
          << "' is not associated with a type which is not an EventSetupRecord.\n"
             "Please check your spelling.";
    }

    //now figure out what data
    std::vector<std::pair<std::string, std::string> >& data = it->second;
    if (data.empty()) {
      //get everything from the record
      auto keys = iInfo.keysForRecord(rKey);
      for (auto itKey = keys.first, itKeyEnd = keys.second; itKey != itKeyEnd; ++itKey) {
        data.push_back(std::make_pair(std::string(itKey->type().name()), std::string(itKey->name().value())));
      }
    }

    std::vector<DataInfo> dataInfos;
    for (std::vector<std::pair<std::string, std::string> >::iterator itData = data.begin(), itDataEnd = data.end();
         itData != itDataEnd;
         ++itData) {
      HCTypeTag tt = HCTypeTag::findType(itData->first);
      if (tt == HCTypeTag()) {
        throw cms::Exception("UnknownESDataType")
            << "The name '" << itData->first << "' is not associated with a known type held in the " << it->first
            << " Record.\n"
               "Please check spelling or load a module known to link with the package which declares that type.";
      }
      if (!bool(edm::TypeWithDict(tt.value()))) {
        throw cms::Exception("NoDictionary")
            << "The type '" << itData->first << "' can not be retrieved from the Record " << it->first
            << " and stored \n"
               "because no dictionary exists for the type.";
      }
      dataInfos.push_back(DataInfo(tt, itData->second));
      dataInfos.back().m_runToken =
          esConsumes<edm::Transition::BeginRun>(rKey, edm::eventsetup::DataKey(tt, itData->second.c_str()));
      dataInfos.back().m_lumiToken =
          esConsumes<edm::Transition::BeginLuminosityBlock>(rKey, edm::eventsetup::DataKey(tt, itData->second.c_str()));
    }
    m_handlers.push_back(std::make_shared<RecordHandler>(rKey, m_file, dataInfos));
  }
}

FWLiteESRecordWriterAnalyzer::~FWLiteESRecordWriterAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  m_file->Close();
  delete m_file;
}

//
// member functions
//
void FWLiteESRecordWriterAnalyzer::update(const edm::EventSetup& iSetup, bool isRun) {
  for (std::vector<std::shared_ptr<RecordHandler> >::iterator it = m_handlers.begin(), itEnd = m_handlers.end();
       it != itEnd;
       ++it) {
    (*it)->update(iSetup, isRun);
  }
}

// ------------ method called to for each event  ------------
void FWLiteESRecordWriterAnalyzer::analyze(const edm::Event& /*iEvent*/, const edm::EventSetup& iSetup) {}

// ------------ method called once each job just before starting event loop  ------------
void FWLiteESRecordWriterAnalyzer::beginJob() {}

// ------------ method called once each job just after ending the event loop  ------------
void FWLiteESRecordWriterAnalyzer::endJob() { m_file->Write(); }

void FWLiteESRecordWriterAnalyzer::beginRun(edm::Run const&, edm::EventSetup const& iSetup) { update(iSetup, true); }
void FWLiteESRecordWriterAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const& iSetup) {
  update(iSetup, false);
}

//define this as a plug-in
DEFINE_FWK_MODULE(FWLiteESRecordWriterAnalyzer);

EVENTSETUP_RECORD_REG(fwliteeswriter::FWLWEventSetupRecord);
