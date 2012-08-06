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
// $Id: SiteLocalConfigServiceTester.cc,v 1.1 2010/04/23 18:29:58 chrjones Exp $
//

// system include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/Exception.h"

// user include files

namespace edmtest {
   class SiteLocalConfigServiceTester : public edm::EDAnalyzer {
   public:
      SiteLocalConfigServiceTester(const edm::ParameterSet& iPSet);
      
      void analyze(const edm::Event&, const edm::EventSetup&);
      
   private:
      std::string m_cacheHint;
      std::string m_readHint;
      std::string m_tempDir;
      unsigned int m_ttreeCacheSize;
      std::vector<std::string> m_nativeProtocols;
      bool m_valuesSet;
   };
}

using namespace edmtest;

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
SiteLocalConfigServiceTester::SiteLocalConfigServiceTester(const edm::ParameterSet& iPSet):
m_cacheHint(iPSet.getUntrackedParameter<std::string>("sourceCacheHint")),
m_readHint(iPSet.getUntrackedParameter<std::string>("sourceReadHint")),
m_tempDir(iPSet.getUntrackedParameter<std::string>("sourceTempDir")),
m_ttreeCacheSize(iPSet.getUntrackedParameter<unsigned int>("sourceTTreeCacheSize")),
m_nativeProtocols(iPSet.getUntrackedParameter<std::vector<std::string> >("sourceNativeProtocols")),
m_valuesSet(iPSet.getUntrackedParameter<bool>("sourceValuesSet",true))
{
}

// SiteLocalConfigServiceTester::SiteLocalConfigServiceTester(const SiteLocalConfigServiceTester& rhs)
// {
//    // do actual copying here;
// }

//SiteLocalConfigServiceTester::~SiteLocalConfigServiceTester()
//{
//}

//
// assignment operators
//
// const SiteLocalConfigServiceTester& SiteLocalConfigServiceTester::operator=(const SiteLocalConfigServiceTester& rhs)
// {
//   //An exception safe implementation is
//   SiteLocalConfigServiceTester temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
static void throwNotSet(const char* iName) {
   throw cms::Exception("TestFailure")<<"The value "<<iName<<" should have been set but was not";
}

template <typename T>
static 
void throwWrongValue(const char* iName, const T& iExpected, const T& iRetrieved) {
   throw cms::Exception("TestFailure")<<"The value "<<iName <<" should have been "<<iExpected<<" but instead was "<<iRetrieved;
}

namespace {
   template <typename T>
   void testValue(const char* iName, const T& iExpected, const T* iRetrieved) {
      if(0==iRetrieved) {
         throwNotSet(iName);
      } else if (*iRetrieved != iExpected) {
         throwWrongValue(iName, iExpected, *iRetrieved);
      }
   }

   template <typename T>
   void checkNotSet(const char* iName, const T* iRetrieved) {
      if(0!=iRetrieved) {
         throw cms::Exception("TestFailure")<<"The value "<<iName<<" should not have been set but was set to "<<*iRetrieved;
      }
   }

   void checkNotSet(const char* iName, const std::vector<std::string>* iRetrieved) {
      if(0!=iRetrieved) {
         throw cms::Exception("TestFailure")<<"The value "<<iName<<" should not have been set but was set";
      }
   }
   
}
//
// const member functions
//
void SiteLocalConfigServiceTester::analyze(const edm::Event&, const edm::EventSetup&)
{
   edm::Service<edm::SiteLocalConfig> pConfig;
   if(m_valuesSet) {
      testValue("sourceCacheTempDir",m_tempDir,pConfig->sourceCacheTempDir());
      testValue("sourceCacheHint",m_cacheHint,pConfig->sourceCacheHint());
      testValue("sourceReadHint",m_readHint,pConfig->sourceReadHint());
      testValue("sourceTTreeCacheSize",m_ttreeCacheSize,pConfig->sourceTTreeCacheSize());
      const std::vector<std::string>* protocols = pConfig->sourceNativeProtocols();
      if(0==protocols) {
         throwNotSet("sourceNativeProtocols");
      }
      if (protocols->size() != m_nativeProtocols.size()) {
         throw cms::Exception("TestFailure")<<"The value sourceNativeProtocols has size "
         <<protocols->size()<<" but should have had size "<<m_nativeProtocols.size();
      }
      for (std::vector<std::string>::const_iterator it = protocols->begin(), itEnd = protocols->end(), 
           itExpect = m_nativeProtocols.begin(); it != itEnd; ++it, ++itExpect) {
         testValue("sourceNativeProtocols",*itExpect,&(*it));
      }
   } else {
      checkNotSet("sourceCacheTempDir",pConfig->sourceCacheTempDir());
      checkNotSet("sourceCacheHint",pConfig->sourceCacheHint());
      checkNotSet("sourceReadHint",pConfig->sourceReadHint());
      checkNotSet("sourceTTreeCacheSize",pConfig->sourceTTreeCacheSize());
      checkNotSet("sourceNativeProtocols",pConfig->sourceNativeProtocols());
   }
   
}

//
// static member functions
//

DEFINE_FWK_MODULE(SiteLocalConfigServiceTester);
