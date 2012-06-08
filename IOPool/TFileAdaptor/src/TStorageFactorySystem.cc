#include "IOPool/TFileAdaptor/interface/TStorageFactorySystem.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"

ClassImp(TStorageFactorySystem)


TStorageFactorySystem::TStorageFactorySystem(const char *, Bool_t)
  : TSystem("-StorageFactory", "Storage Factory System"),
    fDirp(0)
{ SetName("StorageFactory"); }

TStorageFactorySystem::TStorageFactorySystem(void)
  : TSystem("-StorageFactory", "Storage Factory System"),
    fDirp(0)
{ SetName("StorageFactory"); }

TStorageFactorySystem::~TStorageFactorySystem(void)
{}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
Int_t
TStorageFactorySystem::MakeDirectory(const char */*name*/)
{
  Error("MakeDirectory", "Unsupported");
  return -1;
}

void *
TStorageFactorySystem::OpenDirectory(const char */*name*/)
{
  Error("OpenDirectory", "Unsupported");
  return 0;
}

void
TStorageFactorySystem::FreeDirectory(void */*dirp*/)
{
  Error("FreeDirectory", "Unsupported");
}

const char *
TStorageFactorySystem::GetDirEntry(void */*dirp*/)
{
  Error("GetDirEntry", "Unsupported");
  return 0;
}

Bool_t
TStorageFactorySystem::AccessPathName(const char *name, EAccessMode /* mode */)
{
  // NB: This return reverse of check(): kTRUE if access *fails*
  return name ? !StorageFactory::get()->check(name) : kTRUE;
}

Int_t
TStorageFactorySystem::Unlink(const char */*name*/)
{
  Error("Unlink", "Unsupported");
  return 1;
}

Int_t
TStorageFactorySystem::GetPathInfo(const char *name, FileStat_t &info)
{
  info.fDev = 0;
  info.fIno = 0;
  info.fMode = 0644;
  info.fUid = 0;
  info.fGid = 0;
  info.fSize = 0;
  info.fMtime = 0;

  IOOffset storageSize;
  if (StorageFactory::get()->check(name, &storageSize))
  {
    info.fSize = storageSize;
    return 0;
  }

  return -1;
}
