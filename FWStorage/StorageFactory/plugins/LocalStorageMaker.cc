#define _GNU_SOURCE 1
#define _FILE_OFFSET_BITS 64
#include "FWStorage/StorageFactory/interface/StorageMaker.h"
#include "FWStorage/StorageFactory/interface/StorageMakerFactory.h"
#include "FWStorage/StorageFactory/interface/StorageFactory.h"
#include "FWStorage/StorageFactory/interface/File.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

namespace edm::storage {
  class LocalStorageMaker : public StorageMaker {
  public:
    std::unique_ptr<Storage> open(const std::string &proto,
                                  const std::string &path,
                                  int mode,
                                  const AuxSettings &) const override {
      const StorageFactory *f = StorageFactory::get();
      StorageFactory::ReadHint readHint = f->readHint();
      StorageFactory::CacheHint cacheHint = f->cacheHint();

      if (readHint != StorageFactory::READ_HINT_UNBUFFERED || cacheHint == StorageFactory::CACHE_HINT_STORAGE)
        mode &= ~IOFlags::OpenUnbuffered;
      else
        mode |= IOFlags::OpenUnbuffered;

      return std::make_unique<File>(path, mode);
    }

    bool check(const std::string & /*proto*/,
               const std::string &path,
               const AuxSettings &,
               IOOffset *size = nullptr) const override {
      struct stat st;
      if (stat(path.c_str(), &st) != 0)
        return false;

      if (size)
        *size = st.st_size;

      return true;
    }

    UseLocalFile usesLocalFile() const override { return UseLocalFile::kCheckFromPath; }
  };
}  // namespace edm::storage

using namespace edm::storage;
DEFINE_EDM_PLUGIN(StorageMakerFactory, LocalStorageMaker, "file");
