#include "IOPool/TFileAdaptor/interface/TFileAdaptor.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "IOPool/TFileAdaptor/interface/TStorageFactoryFile.h"

#include <TROOT.h>
#include <TSystem.h>
#include <TPluginManager.h>
#include <TEnv.h>
#include <TFile.h>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

void  TFileAdaptorParams::init() const {
  const_cast<TFileAdaptorParams*>(this)->pinit();
}

void  TFileAdaptorParams::pinit() {

  if (doCashing)
    {
      cacheSize = cachePageSize = 0;
    }
  
  //   mode:NAME
  //   pick pre-canned configuration where NAME is one of:
  //      default  = adaptor:on stats:on buffering:off caching:off
  //      raw      = adaptor:on stats:on buffering:off caching:off
  
  
  if (mode == "default")
    {
      doStats = true;
      doBuffering = false;
      doCashing = false;
      cacheSize = TStorageFactoryFile::kDefaultCacheSize;
      cachePageSize = TStorageFactoryFile::kDefaultPageSize;
    }
  else if (mode == "raw")
    { 
      doStats = true;
      doBuffering = false;
      doCashing = false;
      cacheSize = 0;
      cachePageSize = 0;
    }
  
}


bool TFileAdaptorParams::native(const char * prot) const {
  return std::find(m_native.begin(),m_native.end(),prot)!=m_native.end();
}




void TFileAdaptor::addFileType (TPluginManager *mgr, const char *type)
{
  mgr->AddHandler ("TFile", 
		   type, 
		   "TStorageFactoryFile", 
		   "TFileAdaptorModule",
		   "TStorageFactoryFile(const char*,Option_t*,const char*,Int_t)"); 
}

void TFileAdaptor::addSystemType (TPluginManager *mgr, const char *type)
{ 
  mgr->AddHandler ("TSystem", 
		   type, 
		   "TStorageFactorySystem", 
		   "TFileAdaptorModule",
		   "TStorageFactorySystem()"); 
}

bool TFileAdaptor::native(const char * prot) const {
  return m_params.native(prot);
}



TFileAdaptor::TFileAdaptor(const TFileAdaptorParams& iparams): 
  m_params(iparams)
{ 
  std::cerr << "TFileAdaptor loaded" << std::endl;
  
  m_params.init();

  // enable file access stats accounting if requested
  StorageFactory::get()->enableAccounting (m_params.doStats);
  
  // enable file access caching in ROOT if requested
  TStorageFactoryFile::DefaultBuffering (m_params.doBuffering);
  TStorageFactoryFile::DefaultCaching (m_params.cacheSize, m_params.cachePageSize);
  
  // set our own root plugins
  //Plugin.TFile:  ^file: TStorageFactoryFile TFileAdaptorModule "TStorageFactoryFile(const char*,Option_t*,const char*,Int_t)"
  TPluginManager *mgr = gROOT->GetPluginManager();
  if (!native("file")) {addFileType (mgr, "^file:");       addSystemType (mgr, "^file:");}
  if (!native("http")) {addFileType (mgr, "^http:");       addSystemType (mgr, "^http:");}
  if (!native("ftp")) {addFileType (mgr, "^ftp:");        addSystemType (mgr, "^ftp:");}
  {addFileType (mgr, "^web:");        addSystemType (mgr, "^web:");}
  {addFileType (mgr, "^gsiftp:");     addSystemType (mgr, "^gsiftp:");}
  {addFileType (mgr, "^sfn:");        addSystemType (mgr, "^sfn:");}
  {addFileType (mgr, "^zip-member:"); addSystemType (mgr, "^zip-member:");}
  if (!native("rfio")){addFileType (mgr, "^rfio:");       addSystemType (mgr, "^rfio:");}
  if (!native("dcache")){addFileType (mgr, "^dcache:");     addSystemType (mgr, "^dcache:");}
  if (!native("dcap")){addFileType (mgr, "^dcap:");       addSystemType (mgr, "^dcap:");}
  if (!native("gsicdap")) {addFileType (mgr, "^gsidcap:");    addSystemType (mgr, "^gsidcap:");}
   
}
//                  gROOT->GetPluginManager()->Print(); // use option="a" to see ctors 

TFileAdaptor::~TFileAdaptor () {}

void TFileAdaptor::stats(std::ostream& co) const
{
  co << "\n\n"
     << "Storage parameters: adaptor: true"
     << " Stats:" << (m_params.doStats ? "true" : "false")
     << " Buffering:" << (m_params.doBuffering ? "true" : "false")
     << " Caching:" << m_params.cacheSize << "," << m_params.cachePageSize << '\n';
  
  co << "Storage statistics: "
     << StorageAccount::summaryText ()
     << "; tfile/read=?/?/" << (TFile::GetFileBytesRead () / 1048576.0) << "MB/?ms/?ms/?ms"
     << "; tfile/write=?/?/" << (TFile::GetFileBytesWritten () / 1048576.0) << "MB/?ms/?ms/?ms\n"
     << "\n\n";
}
