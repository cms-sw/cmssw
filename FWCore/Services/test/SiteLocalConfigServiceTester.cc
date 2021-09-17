// -*- C++ -*-
//
// Package:     Services
// Class  :     SiteLocalConfigServiceTester
//
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Tue Apr 20 16:51:38 CDT 2010
//

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <string>

namespace edmtest {
  class SiteLocalConfigServiceTester : public edm::EDAnalyzer {
  public:
    SiteLocalConfigServiceTester(const edm::ParameterSet& iPSet);

    void analyze(const edm::Event&, const edm::EventSetup&) override;

  private:
    std::string m_cacheHint;
    std::string m_readHint;
    std::string m_tempDir;
    unsigned int m_ttreeCacheSize;
    std::vector<std::string> m_nativeProtocols;
    bool m_valuesSet;
    bool m_expectedUseLocalConnectString;
    std::string m_expectedLocalConnectPrefix;
    std::string m_expectedLocalConnectSuffix;
  };
}  // namespace edmtest

using namespace edmtest;

SiteLocalConfigServiceTester::SiteLocalConfigServiceTester(const edm::ParameterSet& iPSet)
    : m_cacheHint(iPSet.getUntrackedParameter<std::string>("sourceCacheHint")),
      m_readHint(iPSet.getUntrackedParameter<std::string>("sourceReadHint")),
      m_tempDir(iPSet.getUntrackedParameter<std::string>("sourceTempDir")),
      m_ttreeCacheSize(iPSet.getUntrackedParameter<unsigned int>("sourceTTreeCacheSize")),
      m_nativeProtocols(iPSet.getUntrackedParameter<std::vector<std::string> >("sourceNativeProtocols")),
      m_valuesSet(iPSet.getUntrackedParameter<bool>("sourceValuesSet", true)),
      m_expectedUseLocalConnectString(iPSet.getUntrackedParameter<bool>("expectedUseLocalConnectString")),
      m_expectedLocalConnectPrefix(iPSet.getUntrackedParameter<std::string>("expectedLocalConnectPrefix")),
      m_expectedLocalConnectSuffix(iPSet.getUntrackedParameter<std::string>("expectedLocalConnectSuffix")) {}

static void throwNotSet(const char* iName) {
  throw cms::Exception("TestFailure") << "The value " << iName << " should have been set but was not";
}

template <typename T>
static void throwWrongValue(const char* iName, const T& iExpected, const T& iRetrieved) {
  throw cms::Exception("TestFailure") << "The value " << iName << " should have been " << iExpected
                                      << " but instead was " << iRetrieved;
}

namespace {
  template <typename T>
  void testValue(const char* iName, const T& iExpected, const T* iRetrieved) {
    if (nullptr == iRetrieved) {
      throwNotSet(iName);
    } else if (*iRetrieved != iExpected) {
      throwWrongValue(iName, iExpected, *iRetrieved);
    }
  }

  template <typename T>
  void checkNotSet(const char* iName, const T* iRetrieved) {
    if (nullptr != iRetrieved) {
      throw cms::Exception("TestFailure")
          << "The value " << iName << " should not have been set but was set to " << *iRetrieved;
    }
  }

  void checkNotSet(const char* iName, const std::vector<std::string>* iRetrieved) {
    if (nullptr != iRetrieved) {
      throw cms::Exception("TestFailure") << "The value " << iName << " should not have been set but was set";
    }
  }

}  // namespace

void SiteLocalConfigServiceTester::analyze(const edm::Event&, const edm::EventSetup&) {
  edm::Service<edm::SiteLocalConfig> pConfig;
  if (m_valuesSet) {
    testValue("sourceCacheTempDir", m_tempDir, pConfig->sourceCacheTempDir());
    testValue("sourceCacheHint", m_cacheHint, pConfig->sourceCacheHint());
    testValue("sourceReadHint", m_readHint, pConfig->sourceReadHint());
    testValue("sourceTTreeCacheSize", m_ttreeCacheSize, pConfig->sourceTTreeCacheSize());
    const std::vector<std::string>* protocols = pConfig->sourceNativeProtocols();
    if (nullptr == protocols) {
      throwNotSet("sourceNativeProtocols");
    }
    if (protocols->size() != m_nativeProtocols.size()) {
      throw cms::Exception("TestFailure") << "The value sourceNativeProtocols has size " << protocols->size()
                                          << " but should have had size " << m_nativeProtocols.size();
    }
    for (std::vector<std::string>::const_iterator it = protocols->begin(),
                                                  itEnd = protocols->end(),
                                                  itExpect = m_nativeProtocols.begin();
         it != itEnd;
         ++it, ++itExpect) {
      testValue("sourceNativeProtocols", *itExpect, &(*it));
    }
  } else {
    checkNotSet("sourceCacheTempDir", pConfig->sourceCacheTempDir());
    checkNotSet("sourceCacheHint", pConfig->sourceCacheHint());
    checkNotSet("sourceReadHint", pConfig->sourceReadHint());
    checkNotSet("sourceTTreeCacheSize", pConfig->sourceTTreeCacheSize());
    checkNotSet("sourceNativeProtocols", pConfig->sourceNativeProtocols());
  }

  if (pConfig->useLocalConnectString() != m_expectedUseLocalConnectString) {
    throw cms::Exception("TestFailure") << "The value of useLocalConnectString is \""
                                        << (pConfig->useLocalConnectString() ? std::string("true")
                                                                             : std::string("false"))
                                        << "\" but we expected the value \""
                                        << (m_expectedUseLocalConnectString ? std::string("true")
                                                                            : std::string("false"))
                                        << "\"";
  }
  if (pConfig->localConnectPrefix() != m_expectedLocalConnectPrefix) {
    throw cms::Exception("TestFailure") << "The value of localConnectPrefix is \"" << pConfig->localConnectPrefix()
                                        << "\" but we expected the value \"" << m_expectedLocalConnectPrefix << "\"";
  }
  if (pConfig->localConnectSuffix() != m_expectedLocalConnectSuffix) {
    throw cms::Exception("TestFailure") << "The value of localConnectSuffix is \"" << pConfig->localConnectSuffix()
                                        << "\" but we expected the value \"" << m_expectedLocalConnectSuffix << "\"";
  }
}

DEFINE_FWK_MODULE(SiteLocalConfigServiceTester);
