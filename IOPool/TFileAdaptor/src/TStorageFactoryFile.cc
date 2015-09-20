#include "IOPool/TFileAdaptor/interface/TStorageFactoryFile.h"
#include "Utilities/StorageFactory/interface/Storage.h"
#include "Utilities/StorageFactory/interface/StorageFactory.h"
#include "Utilities/StorageFactory/interface/StorageAccount.h"
#include "Utilities/StorageFactory/interface/StatisticsSenderService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/ExceptionPropagate.h"
#include "ReadRepacker.h"
#include "TFileCacheRead.h"
#include "TSystem.h"
#include "TROOT.h"
#include "TEnv.h"
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <iostream>
#include <cassert>

#if 0
#include "TTreeCache.h"
#include "TTree.h"

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
static StorageAccount::Counter *s_statsWrite = 0;
static StorageAccount::Counter *s_statsCWrite = 0;
static StorageAccount::Counter *s_statsXWrite = 0;


static inline StorageAccount::Counter &
storageCounter(StorageAccount::Counter *&c, StorageAccount::Operation operation)
{
  static const auto token = StorageAccount::tokenForStorageClassName("tstoragefile");
  if (! c) c = &StorageAccount::counter(token, operation);
  return *c;
}

TStorageFactoryFile::TStorageFactoryFile(void)
  : storage_()
{
  StorageAccount::Stamp stats(storageCounter(s_statsCtor, StorageAccount::Operation::construct));
  stats.tick(0);
}

// This constructor must be compatible with *all* the various built-in TFile plugins,
// including TXNetFile.  This is why some arguments in the constructor is ignored.
// If there's a future T*File that is incompatible with this constructor, a new
// constructor will have to be added.
TStorageFactoryFile::TStorageFactoryFile(const char *path,
                                         Option_t *option,
                                         const char *ftitle,
                                         Int_t compress,
                                         Int_t netopt,
                                         Bool_t parallelopen /* = kFALSE */)
  : TFile(path, "NET", ftitle, compress), // Pass "NET" to prevent local access in base class
    storage_()
{
  try {
    Initialize(path, option);
  } catch (...) {
    edm::threadLocalException::setException(std::current_exception()); // capture
  }
}

TStorageFactoryFile::TStorageFactoryFile(const char *path,
                                         Option_t *option /* = "" */,
                                         const char *ftitle /* = "" */,
                                         Int_t compress /* = 1 */)
  : TFile(path, "NET", ftitle, compress), // Pass "NET" to prevent local access in base class
    storage_()
{
  try {
    Initialize(path, option);
  } catch (...) {
    edm::threadLocalException::setException(std::current_exception()); // capture
  }
}

void
TStorageFactoryFile::Initialize(const char *path,
                                Option_t *option /* = "" */)
{
  StorageAccount::Stamp stats(storageCounter(s_statsCtor, StorageAccount::Operation::construct));

  // Enable AsyncReading.
  // This was the default for 5.27, but turned off by default for 5.32.
  // In our testing, AsyncReading is the fastest mechanism available.
  // In 5.32, the AsyncPrefetching mechanism is preferred, but has been a
  // performance hit in our "average case" tests.
  gEnv->SetValue("TFile.AsyncReading", 1);

  // Parse options; at the moment we only accept read!
  fOption = option;
  fOption.ToUpper();

  if (fOption == "NEW")
    fOption = "CREATE";

  Bool_t create   = (fOption == "CREATE");
  Bool_t recreate = (fOption == "RECREATE");
  Bool_t update   = (fOption == "UPDATE");
  Bool_t read     = (fOption == "READ") || (fOption == "READWRAP");
  Bool_t readwrap = (fOption == "READWRAP");

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
  assert(!recreate);

  if (update && gSystem->AccessPathName(path, kFileExists))
  {
    update = kFALSE;
    create = kTRUE;
  }

  assert(read || update || create);

  int           openFlags = IOFlags::OpenRead;
  if (!read)    openFlags |= IOFlags::OpenWrite;
  if (create)   openFlags |= IOFlags::OpenCreate;
  //if (recreate) openFlags |= IOFlags::OpenCreate | IOFlags::OpenTruncate;
  if (readwrap) openFlags |= IOFlags::OpenWrap;

  // Open storage
  if (! (storage_ = StorageFactory::get()->open(path, openFlags)))
  {
     MakeZombie();
     gDirectory = gROOT;
     throw cms::Exception("TStorageFactoryFile::TStorageFactoryFile()")
       << "Cannot open file '" << path << "'";
  }

  // Record the statistics.
  try {
    edm::Service<edm::storage::StatisticsSenderService> statsService;
    if (statsService.isAvailable()) {
      statsService->setSize(storage_->size());
    }
  } catch (edm::Exception e) {
    if (e.categoryCode() != edm::errors::NotFound) {
      throw;
    }
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
}

//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

Bool_t
TStorageFactoryFile::ReadBuffer(char *buf, Long64_t pos, Int_t len)
{
  // This function needs to be optimized to minimize seeks.
  // See TFile::ReadBuffer(char *buf, Long64_t pos, Int_t len) in ROOT 5.27.06.
  Seek(pos);
  return ReadBuffer(buf, len);
}

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
  StorageAccount::Stamp stats(storageCounter(s_statsRead, StorageAccount::Operation::read));

  // If we have a cache, read from there first.  This returns 0
  // if the block hasn't been prefetched, 1 if it was in cache,
  // and 2 if there was an error.
  if (TFileCacheRead *c = GetCacheRead())
  {
    Long64_t here = GetRelOffset();
    Bool_t   async = c->IsAsyncReading();

    StorageAccount::Stamp cstats(async
                                 ? storageCounter(s_statsCPrefetch, StorageAccount::Operation::readPrefetchToCache)
                                 : storageCounter(s_statsCRead, StorageAccount::Operation::readViaCache));

    Int_t st = ReadBufferViaCache(async ? 0 : buf, len);

    if (st == 2) {
      return kTRUE;
    }

    if (st == 1) {
      if (async) {
        cstats.tick(len);
        Seek(here);
      } else {
        cstats.tick(len);
        stats.tick(len);
        return kFALSE;
      }
    }
  }

  // FIXME: Re-enable read-ahead if the data wasn't in cache.
  // if (! st) storage_->caching(true, -1, s_readahead);

  // A real read
  StorageAccount::Stamp xstats(storageCounter(s_statsXRead, StorageAccount::Operation::readActual));
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

  StorageAccount::Stamp stats(storageCounter(s_statsARead, StorageAccount::Operation::readAsync));

  // If asynchronous reading is disabled, bail out now, regardless
  // whether the underlying storage supports prefetching.  If it is
  // forced on, pretend it's on, even if the storage doesn't support
  // it, as this turns off the caching in ROOT's side.
  const StorageFactory *f = StorageFactory::get();

  // Verify that we never using async reads in app-only mode
  if (f->cacheHint() == StorageFactory::CACHE_HINT_APPLICATION)
    return kTRUE;

  // Let the I/O method indicate if it can do client-side prefetch.
  // If it does, then for example TTreeCache will drop its own cache
  // and will use the client-side cache of the actual I/O layer.
  // If len is zero ROOT is probing for prefetch support.
  if (len) {
    // FIXME: Synchronise caching.
    // storage_->caching(true, -1, 0);
    ;
  }

  IOPosBuffer iov(off, (void *) 0, len ? len : PREFETCH_PROBE_LENGTH);
  if (storage_->prefetch(&iov, 1))
  {
    stats.tick(len);
    return kFALSE;
  }

  // Always ask ROOT to use async reads in storage-only mode,
  // regardless of whether the storage system supports it.
  if (f->cacheHint() == StorageFactory::CACHE_HINT_STORAGE)
    return kFALSE;

  // Prefetching not available right now.
  return kTRUE;
}

Bool_t
TStorageFactoryFile::ReadBuffersSync(char *buf, Long64_t *pos, Int_t *len, Int_t nbuf)
{
  /** Most storage systems are not prepared for the onslaught of small reads
   *  that ROOT will perform, even if they implement a vectored read interface.
   *
   *  Typically, on the server side, the loop is unrolled and the reads are
   *  issued sequentially - giving the OS no hint that you're about to read
   *  a very close-by byte in the near future.  Normally, OS read-ahead takes
   *  care of such situations; because the storage server has so many clients,
   *  and ROOT reads look random to the OS, the read-ahead becomes disabled.
   *
   *  Hence, this function will repack the application-layer request into an
   *  optimized storage-layer request.  The resulting request to the storage
   *  layer typically has a slightly larger number of bytes, but far less
   *  individual reads.
   *
   *  On average, the server's disks see a smaller number of overall reads,
   *  the number of bytes transferred over the network increases modestly
   *  (around 10%), and the single application request becomes one-to-two
   *  I/O transactions.  A clear win for all cases except high-latency WAN.
   */

  Int_t remaining = nbuf; // Number of read requests left to process.
  Int_t pack_count; // Number of read requests processed by this iteration.

  IOSize remaining_buffer_size=0;
  // Calculate the remaining buffer size for the ROOT-owned buffer by adding
  // the size of the various requests.
  for (Int_t i=0; i<nbuf; i++) remaining_buffer_size+=len[i];

  char     *current_buffer = buf;
  Long64_t *current_pos    = pos;
  Int_t    *current_len    = len;

  ReadRepacker repacker;

  while (remaining > 0) {

    pack_count = repacker.pack(static_cast<long long int *>(current_pos), current_len, remaining, current_buffer, remaining_buffer_size);

    int real_bytes_processed = repacker.realBytesProcessed();
    IOSize io_buffer_used = repacker.bufferUsed();

    // Issue readv, then unpack buffers.
    StorageAccount::Stamp xstats(storageCounter(s_statsXRead, StorageAccount::Operation::readActual));
    std::vector<IOPosBuffer> &iov = repacker.iov();
    IOSize result = storage_->readv(&iov[0], iov.size());
    if (result != io_buffer_used) {
      return kTRUE;
    }
    xstats.tick(io_buffer_used);
    repacker.unpack(current_buffer);

    // Update the location of the unused part of the input buffer.
    remaining_buffer_size -= real_bytes_processed;
    current_buffer += real_bytes_processed;

    current_pos += pack_count;
    current_len += pack_count;
    remaining   -= pack_count;

  }
  assert(remaining_buffer_size == 0);
  return kFALSE;
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

  // For synchronous reads, we have special logic to optimize the I/O requests
  // from ROOT before handing it to the storage.
  if (buf)
  {
    return ReadBuffersSync(buf, pos, len, nbuf);
  }
  // For an async read, we assume the storage system is smart enough to do the
  // optimization itself.

  // Read from underlying storage.
  void* const nobuf = 0;
  Int_t total = 0;
  std::vector<IOPosBuffer> iov;
  iov.reserve(nbuf);
  for (Int_t i = 0; i < nbuf; ++i)
  {
    iov.push_back(IOPosBuffer(pos[i], nobuf, len[i]));
    total += len[i];
  }

  // Null buffer means asynchronous reads into I/O system's cache.
  bool success;
  StorageAccount::Stamp astats(storageCounter(s_statsARead, StorageAccount::Operation::readAsync));
  // Synchronise low-level cache with the supposed cache in TFile.
  // storage_->caching(true, -1, 0);
  success = storage_->prefetch(&iov[0], nbuf);
  astats.tick(total);

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

  StorageAccount::Stamp stats(storageCounter(s_statsWrite, StorageAccount::Operation::write));
  StorageAccount::Stamp cstats(storageCounter(s_statsCWrite, StorageAccount::Operation::writeViaCache));

  // Try first writing via a cache, and if that's not possible, directly.
  Int_t st;
  switch ((st = WriteBufferViaCache(buf, len)))
  {
  case 0:
    // Actual write.
    {
      StorageAccount::Stamp xstats(storageCounter(s_statsXWrite, StorageAccount::Operation::writeActual));
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
TStorageFactoryFile::SysOpen(const char *pathname, Int_t flags, UInt_t /* mode */)
{
  StorageAccount::Stamp stats(storageCounter(s_statsOpen, StorageAccount::Operation::open));

  if (storage_)
  {
    storage_->close();
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
  StorageAccount::Stamp stats(storageCounter(s_statsClose, StorageAccount::Operation::close));

  if (storage_)
  {
    storage_->close();
    storage_.release();
  }

  stats.tick();
  return 0;
}

Long64_t
TStorageFactoryFile::SysSeek(Int_t /* fd */, Long64_t offset, Int_t whence)
{
  StorageAccount::Stamp stats(storageCounter(s_statsSeek, StorageAccount::Operation::seek));
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
  StorageAccount::Stamp stats(storageCounter(s_statsFlush, StorageAccount::Operation::flush));
  storage_->flush();
  stats.tick();
  return 0;
}

Int_t
TStorageFactoryFile::SysStat(Int_t /* fd */, Long_t *id, Long64_t *size,
                             Long_t *flags, Long_t *modtime)
{
  StorageAccount::Stamp stats(storageCounter(s_statsStat, StorageAccount::Operation::stat));
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
