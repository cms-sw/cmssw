#include "IOPool/TFileAdaptor/interface/TStorageFactoryFile.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "TFileCacheRead.h"
#include "TFileCacheWrite.h"
#include "TFileCacheRead.h"
#include "TSystem.h"
#include "TROOT.h"
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

#if 0
#include "TTreeCache.h"
#include "TTree.h"
#include <iostream>

class TTreeCacheDebug : public TTreeCache {
public:
  void dump(const char *label, const char *trailer)
  {
    Long64_t entry = fOwner->GetReadEntry();
    std::cerr
      << label << ": " << entry << " "
      << "{ fEntryMin=" << fEntryMin
      << ", fEntryMax=" << fEntryMax
      << ", fEntryNext=" << fEntryNext
      << ", fZipBytes=" << fZipBytes
      << ", fNbranches=" << fNbranches
      << ", fNReadOk=" << fNReadOk
      << ", fNReadMiss=" << fNReadMiss
      << ", fNReadPref=" << fNReadPref
      << ", fBranches=" << fBranches
      << ", fBrNames=" << fBrNames
      << ", fOwner=" << fOwner
      << ", fTree=" << fTree
      << ", fIsLearning=" << fIsLearning
      << ", fIsManual=" << fIsManual
      << "; fBufferSizeMin=" << fBufferSizeMin
      << ", fBufferSize=" << fBufferSize
      << ", fBufferLen=" << fBufferLen
      << ", fBytesToPrefetch=" << fBytesToPrefetch
      << ", fFirstIndexToPrefetch=" << fFirstIndexToPrefetch
      << ", fAsyncReading=" << fAsyncReading
      << ", fNseek=" << fNseek
      << ", fNtot=" << fNtot
      << ", fNb=" << fNb
      << ", fSeekSize=" << fSeekSize
      << ", fSeek=" << fSeek
      << ", fSeekIndex=" << fSeekIndex
      << ", fSeekSort=" << fSeekSort
      << ", fPos=" << fPos
      << ", fSeekLen=" << fSeekLen
      << ", fSeekSortLen=" << fSeekSortLen
      << ", fSeekPos=" << fSeekPos
      << ", fLen=" << fLen
      << ", fFile=" << fFile
      << ", fBuffer=" << (void *) fBuffer
      << ", fIsSorted=" << fIsSorted
      << " }\n" << trailer;
  }
};
#endif

ClassImp(TStorageFactoryFile)
static StorageAccount::Counter *s_statsCtor = 0;
static StorageAccount::Counter *s_statsOpen = 0;
static StorageAccount::Counter *s_statsClose = 0;
static StorageAccount::Counter *s_statsFlush = 0;
static StorageAccount::Counter *s_statsStat = 0;
static StorageAccount::Counter *s_statsSeek = 0;
static StorageAccount::Counter *s_statsRead = 0;
static StorageAccount::Counter *s_statsCRead = 0;
static StorageAccount::Counter *s_statsCPrefetch = 0;
static StorageAccount::Counter *s_statsARead = 0;
static StorageAccount::Counter *s_statsXRead = 0;
static StorageAccount::Counter *s_statsVRead = 0;
static StorageAccount::Counter *s_statsWrite = 0;
static StorageAccount::Counter *s_statsCWrite = 0;
static StorageAccount::Counter *s_statsXWrite = 0;

static inline StorageAccount::Counter &
storageCounter(StorageAccount::Counter *&c, const char *label)
{
  if (! c) c = &StorageAccount::counter("tstoragefile", label);
  return *c;
}

TStorageFactoryFile::TStorageFactoryFile(void)
  : storage_(0)
{
  StorageAccount::Stamp stats(storageCounter(s_statsCtor, "construct"));
  stats.tick(0);
}


TStorageFactoryFile::TStorageFactoryFile(const char *path,
					 Option_t *option /* = "" */,
					 const char *ftitle /* = "" */,
					 Int_t compress /* = 1 */)
  : TFile(path, "NET", ftitle, compress), // Pass "NET" to prevent local access in base class
    storage_(0)
{
  StorageAccount::Stamp stats(storageCounter(s_statsCtor, "construct"));

  // Parse options; at the moment we only accept read!
  fOption = option;
  fOption.ToUpper();

  if (fOption == "NEW")
    fOption = "CREATE";

  Bool_t create   = (fOption == "CREATE");
  Bool_t recreate = (fOption == "RECREATE");
  Bool_t update   = (fOption == "UPDATE");
  Bool_t read     = (fOption == "READ");

  if (!create && !recreate && !update && !read)
  {
    read = true;
    fOption = "READ";
  }

  if (recreate)
  {
    if (!gSystem->AccessPathName(path, kFileExists))
      gSystem->Unlink(path);

    recreate = false;
    create   = true;
    fOption  = "CREATE";
  }

  if (update && gSystem->AccessPathName(path, kFileExists))
  {
    update = kFALSE;
    create = kTRUE;
  }

  int           openFlags = IOFlags::OpenRead;
  if (!read)    openFlags |= IOFlags::OpenWrite;
  if (create)   openFlags |= IOFlags::OpenCreate;
  if (recreate) openFlags |= IOFlags::OpenCreate | IOFlags::OpenTruncate;

  // Open storage
  if (! (storage_ = StorageFactory::get()->open(path, openFlags)))
  {
     MakeZombie();
     gDirectory = gROOT;
     throw cms::Exception("TStorageFactoryFile::TStorageFactoryFile()")
       << "Cannot open file '" << path << "'";
  }

  fRealName = path;
  fD = 0; // sorry, meaningless
  fWritable = read ? kFALSE : kTRUE;

  Init(create);

  stats.tick(0);
}

TStorageFactoryFile::~TStorageFactoryFile(void)
{
  Close();
  delete storage_;
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
Bool_t
TStorageFactoryFile::ReadBuffer(char *buf, Int_t len)
{
  // Check that it's valid to access this file.
  if (IsZombie())
  {
    Error("ReadBuffer", "Cannot read from a zombie file");
    return kTRUE;
  }

  if (! IsOpen())
  {
    Error("ReadBuffer", "Cannot read from a file that is not open");
    return kTRUE;
  }

  // Read specified byte range from the storage.  Returns kTRUE in
  // case of error.  Note that ROOT uses this function recursively
  // to fill the cache; we use a flag to make sure our accounting
  // is reflected in a comprehensible manner.  The "read" counter
  // will include both, "readc" indicates how much read from the
  // cache, "readu" indicates how much we failed to read from the
  // cache (excluding those recursive reads), and "readx" counts
  // the amount actually passed to read from the storage object.
  StorageAccount::Stamp stats(storageCounter(s_statsRead, "read"));

  // If we have a cache, read from there first.  This returns 0
  // if the block hasn't been prefetched, 1 if it was in cache,
  // and 2 if there was an error.
  if (TFileCacheRead *c = GetCacheRead())
  {
    Long64_t here = GetRelOffset();
    Bool_t   async = c->IsAsyncReading();

    StorageAccount::Stamp cstats(async
				 ? storageCounter(s_statsCPrefetch, "read-prefetch-to-cache")
				 : storageCounter(s_statsCRead, "read-via-cache"));

    Int_t st = ReadBufferViaCache(async ? 0 : buf, len);

    if (st == 2)
      return kTRUE;

    if (st == 1)
      if (async)
      {
        cstats.tick(len);
	Seek(here);
      }
      else
      {
        cstats.tick(len);
        stats.tick(len);
        return kFALSE;
      }
  }

  // FIXME: Re-enable read-ahead if the data wasn't in cache.
  // if (! st) storage_->caching(true, -1, s_readahead);

  // A real read
  StorageAccount::Stamp xstats(storageCounter(s_statsXRead, "read-actual"));
  IOSize n = storage_->xread(buf, len);
  xstats.tick(n);
  stats.tick(n);
  return n ? kFALSE : kTRUE;
}

Bool_t
TStorageFactoryFile::ReadBufferAsync(Long64_t off, Int_t len)
{
  // Check that it's valid to access this file.
  if (IsZombie())
  {
    Error("ReadBufferAsync", "Cannot read from a zombie file");
    return kTRUE;
  }

  if (! IsOpen())
  {
    Error("ReadBufferAsync", "Cannot read from a file that is not open");
    return kTRUE;
  }

  StorageAccount::Stamp stats(storageCounter(s_statsARead, "read-async"));

  // If asynchronous reading is disabled, bail out now, regardless
  // whether the underlying storage supports prefetching.  If it is
  // forced on, pretend it's on, even if the storage doesn't support
  // it, as this turns off the caching in ROOT's side.
  StorageFactory *f = StorageFactory::get();
  switch (f->cacheHint())
  {
  case StorageFactory::CACHE_HINT_APPLICATION:
    return kTRUE;
  case StorageFactory::CACHE_HINT_STORAGE:
    return kFALSE;
  default:
    break;
  }

  // Let the I/O method indicate if it can do client-side prefetch.
  // If it does, then for example TTreeCache will drop its own cache
  // and will use the client-side cache of the actual I/O layer.
  // If len is zero ROOT is probing for prefetch support.
  if (len)
    // FIXME: Synchronise caching.
    // storage_->caching(true, -1, 0);
    ;

  IOPosBuffer iov(off, (void *) 0, len ? len : 4096);
  if (storage_->prefetch(&iov, 1))
  {
    stats.tick(len);
    return kFALSE;
  }

  // Prefetching not available right now.
  return kTRUE;
}

Bool_t
TStorageFactoryFile::ReadBuffers(char *buf, Long64_t *pos, Int_t *len, Int_t nbuf)
{
  // Check that it's valid to access this file.
  if (IsZombie())
  {
    Error("ReadBuffers", "Cannot read from a zombie file");
    return kTRUE;
  }

  if (! IsOpen())
  {
    Error("ReadBuffers", "Cannot read from a file that is not open");
    return kTRUE;
  }

  // Read from underlying storage.
  Int_t total = 0;
  std::vector<IOPosBuffer> iov;
  iov.reserve(nbuf);
  for (Int_t i = 0; i < nbuf; ++i)
  {
    iov.push_back(IOPosBuffer(pos[i], buf ? buf + total : 0, len[i]));
    total += len[i];
  }

  // Null buffer means asynchronous reads into I/O system's cache.
  bool success;
  if (buf)
  {
    StorageAccount::Stamp stats(storageCounter(s_statsVRead, "readv-actual"));
    IOSize n = storage_->readv(&iov[0], nbuf);
    success = (n > 0); // FIXME: what if it's short!?
    stats.tick(n);
  }
  else
  {
    StorageAccount::Stamp astats(storageCounter(s_statsARead, "read-async"));
    // Synchronise low-level cache with the supposed cache in TFile.
    // storage_->caching(true, -1, 0);
    success = storage_->prefetch(&iov[0], nbuf);
    astats.tick(total);
  }

  // If it didn't suceeed, pass down to the base class.
  return success ? kFALSE : TFile::ReadBuffers(buf, pos, len, nbuf);
}

Bool_t
TStorageFactoryFile::WriteBuffer(const char *buf, Int_t len)
{
  // Check that it's valid to access this file.
  if (IsZombie())
  {
    Error("WriteBuffer", "Cannot write to a zombie file");
    return kTRUE;
  }

  if (! IsOpen())
  {
    Error("WriteBuffer", "Cannot write to a file that is not open");
    return kTRUE;
  }

  if (! fWritable)
  {
    Error("WriteBuffer", "File is not writable");
    return kTRUE;
  }

  StorageAccount::Stamp stats(storageCounter(s_statsWrite, "write"));
  StorageAccount::Stamp cstats(storageCounter(s_statsCWrite, "write-via-cache"));

  // Try first writing via a cache, and if that's not possible, directly.
  Int_t st;
  switch ((st = WriteBufferViaCache(buf, len)))
  {
  case 0:
    // Actual write.
    {
      StorageAccount::Stamp xstats(storageCounter(s_statsXWrite, "write-actual"));
      IOSize n = storage_->xwrite(buf, len);
      xstats.tick(n);
      stats.tick(n);

      // FIXME: What if it's a short write?
      return n > 0 ? kFALSE : kTRUE;
    }

  case 1:
    cstats.tick(len);
    stats.tick(len);
    return kFALSE;

  case 2:
  default:
    Error("WriteBuffer", "Error writing to cache");
    return kTRUE;
  }
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// FIXME: Override GetBytesToPrefetch() so XROOTD can suggest how
// large a prefetch cache to use.
// FIXME: Asynchronous open support?

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
Int_t
TStorageFactoryFile::SysOpen(const char *pathname, Int_t flags, UInt_t mode)
{
  StorageAccount::Stamp stats(storageCounter(s_statsOpen, "open"));

  if (storage_)
  {
    storage_->close();
    delete storage_;
    storage_ = 0;
  }

  int                      openFlags = IOFlags::OpenRead;
  if (flags & O_WRONLY)    openFlags = IOFlags::OpenWrite;
  else if (flags & O_RDWR) openFlags |= IOFlags::OpenWrite;
  if (flags & O_CREAT)     openFlags |= IOFlags::OpenCreate;
  if (flags & O_APPEND)    openFlags |= IOFlags::OpenAppend;
  if (flags & O_EXCL)      openFlags |= IOFlags::OpenExclusive;
  if (flags & O_TRUNC)     openFlags |= IOFlags::OpenTruncate;
  if (flags & O_NONBLOCK)  openFlags |= IOFlags::OpenNonBlock;

  if (! (storage_ = StorageFactory::get()->open(pathname, openFlags)))
  {
     MakeZombie();
     gDirectory = gROOT;
     throw cms::Exception("TStorageFactoryFile::SysOpen()")
       << "Cannot open file '" << pathname << "'";
  }

  stats.tick();
  return 0;
}

Int_t
TStorageFactoryFile::SysClose(Int_t /* fd */)
{
  StorageAccount::Stamp stats(storageCounter(s_statsClose, "close"));

  if (storage_)
  {
    storage_->close();
    delete storage_;
    storage_ = 0;
  }

  stats.tick();
  return 0;
}

Long64_t
TStorageFactoryFile::SysSeek(Int_t /* fd */, Long64_t offset, Int_t whence)
{
  StorageAccount::Stamp stats(storageCounter(s_statsSeek, "seek"));
  Storage::Relative rel = (whence == SEEK_SET ? Storage::SET
    	    	           : whence == SEEK_CUR ? Storage::CURRENT
    		           : Storage::END);

  offset = storage_->position(offset, rel);
  stats.tick();
  return offset;
}

Int_t
TStorageFactoryFile::SysSync(Int_t /* fd */)
{
  StorageAccount::Stamp stats(storageCounter(s_statsFlush, "flush"));
  storage_->flush();
  stats.tick();
  return 0;
}

Int_t
TStorageFactoryFile::SysStat(Int_t /* fd */, Long_t *id, Long64_t *size,
    		      Long_t *flags, Long_t *modtime)
{
  StorageAccount::Stamp stats(storageCounter(s_statsStat, "stat"));
  // FIXME: Most of this is unsupported or makes no sense with Storage
  *id = ::Hash(fRealName);
  *size = storage_->size();
  *flags = 0;
  *modtime = 0;
  stats.tick();
  return 0;
}

void
TStorageFactoryFile::ResetErrno(void) const
{
  TSystem::ResetErrno();
}
