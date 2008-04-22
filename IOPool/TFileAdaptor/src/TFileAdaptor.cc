#include "IOPool/TFileAdaptor/interface/TFileAdaptor.h"
#include "IOPool/TFileAdaptor/interface/TStorageFactoryFile.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include <TROOT.h>
#include <TPluginManager.h>
#include <TFile.h>
#include <iostream>
#include <algorithm>

void
TFileAdaptorParams::init (void) const
{
  const_cast<TFileAdaptorParams*>(this)->pinit();
}

void
TFileAdaptorParams::pinit (void)
{
  if (doCaching)
    cacheSize = cachePageSize = 0;

  //   mode:NAME
  //   pick pre-canned configuration where NAME is one of:
  //      default  = adaptor:on stats:on buffering:off caching:off
  //      raw      = adaptor:on stats:on buffering:off caching:off

  if (mode == "default")
  {
    doStats = true;
    doBuffering = false;
    doCaching = false;
    cacheSize = TStorageFactoryFile::kDefaultCacheSize;
    cachePageSize = TStorageFactoryFile::kDefaultPageSize;
  }
  else if (mode == "raw")
  { 
    doStats = true;
    doBuffering = false;
    doCaching = false;
    cacheSize = 0;
    cachePageSize = 0;
  }
}

bool
TFileAdaptorParams::native (const char *prot) const
{
  return std::find(m_native.begin(), m_native.end(), "all") != m_native.end()
    || std::find(m_native.begin(), m_native.end(), prot) != m_native.end();
}

void
TFileAdaptor::addType (TPluginManager *mgr, const char *type)
{
  mgr->AddHandler ("TFile", 
		   type, 
		   "TStorageFactoryFile", 
		   "IOPoolTFileAdaptor",
		   "TStorageFactoryFile(const char*,Option_t*,const char*,Int_t)"); 

  mgr->AddHandler ("TSystem", 
		   type, 
		   "TStorageFactorySystem", 
		   "IOPoolTFileAdaptor",
		   "TStorageFactorySystem()"); 
}

bool
TFileAdaptor::native (const char *prot) const
{
  return m_params.native(prot);
}

TFileAdaptor::TFileAdaptor (const TFileAdaptorParams& iparams)
  : m_params(iparams)
{ 
  m_params.init();

  // enable file access stats accounting if requested
  StorageFactory::get()->enableAccounting (m_params.doStats);

  // enable file access caching in ROOT if requested
  TStorageFactoryFile::DefaultBuffering (m_params.doBuffering);
  TStorageFactoryFile::DefaultCaching (m_params.cacheSize, m_params.cachePageSize);

  // set our own root plugins
  TPluginManager *mgr = gROOT->GetPluginManager();

  // Load the ROOT plugins first so that we will overwrite them
  // instead of them overwriting our plugins later
  mgr->LoadHandlersFromPluginDirs("TFile");
  mgr->LoadHandlersFromPluginDirs("TSystem");

  if (!native("file"))    addType (mgr, "^file:");
  if (!native("http"))    addType (mgr, "^http:");
  if (!native("ftp"))     addType (mgr, "^ftp:");
  /* always */            addType (mgr, "^web:");
  /* always */            addType (mgr, "^gsiftp:");
  /* always */            addType (mgr, "^sfn:");
  if (!native("rfio"))    addType (mgr, "^rfio:");
  if (!native("dcache"))  addType (mgr, "^dcache:");
  if (!native("dcap"))    addType (mgr, "^dcap:");
  if (!native("gsidcap")) addType (mgr, "^gsidcap:");
}

TFileAdaptor::~TFileAdaptor (void)
{}

void
TFileAdaptor::stats (std::ostream &o) const
{
  if (! m_params.doStats)
    return;

  o << "Storage parameters: adaptor: true"
    << " Stats:" << (m_params.doStats ? "true" : "false")
    << " Buffering:" << (m_params.doBuffering ? "true" : "false")
    << " Caching:" << m_params.cacheSize << "," << m_params.cachePageSize << '\n'

    << "Storage statistics: "
    << StorageAccount::summaryText ()
    << "; tfile/read=?/?/" << (TFile::GetFileBytesRead () / 1048576.0) << "MB/?ms/?ms/?ms"
    << "; tfile/write=?/?/" << (TFile::GetFileBytesWritten () / 1048576.0) << "MB/?ms/?ms/?ms";
}

void
TFileAdaptor::statsXML (std::ostream &o) const
{
  if (! m_params.doStats)
    return;

  o << "<storage-factory-summary>\n"
    << " <storage-factory-params>\n"
    << "  <param name='enabled' value='true' unit='boolean'/>\n"
    << "  <param name='stats' value='" << (m_params.doStats ? "true" : "false") << "' unit='boolean'/>\n"
    << "  <param name='buffering' value='" << (m_params.doBuffering ? "true" : "false") << "' unit='boolean'/>\n"
    << "  <param name='cache-size' value='" << (m_params.cacheSize * 1048576) << "' unit='bytes'/>\n"
    << "  <param name='cache-pages' value='" << m_params.cachePageSize << "' unit='bytes'/>\n"
    << " </storage-factory-params>\n"

    << " <storage-factory-stats>\n"
    << StorageAccount::summaryXML () << std::endl
    << "  <storage-root-summary>\n"
    << "   <counter-value subsystem='tfile' counter-name='read' total-megabytes='"
    << (TFile::GetFileBytesRead () / 1048576.0) << "'/>\n"
    << "   <counter-value subsystem='tfile' counter-name='write' total-megabytes='"
    << (TFile::GetFileBytesWritten() / 1048576.0) << "'/>\n"
    << "  </storage-root-summary>\n"
    << " </storage-factory-stats>\n"
    << "</storage-factory-summary>";
}
