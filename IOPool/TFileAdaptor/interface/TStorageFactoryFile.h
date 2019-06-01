#ifndef TFILE_ADAPTOR_TSTORAGE_FACTORY_FILE_H
#define TFILE_ADAPTOR_TSTORAGE_FACTORY_FILE_H

#include <vector>
#include <memory>

#include "TFile.h"

#include "Utilities/StorageFactory/interface/IOPosBuffer.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

class Storage;

/** TFile wrapper around #StorageFactory and #Storage.  */
class TStorageFactoryFile : public TFile {
public:
  ClassDefOverride(TStorageFactoryFile, 0);  // ROOT File operating on CMS Storage.

  // Due to limitations in the ROOT plugin manager, TStorageFactoryFile must
  // provide a constructor matching all the different variants that other
  // ROOT plugins might use

  // This one is to match TXNetFile
  TStorageFactoryFile(const char *name,
                      Option_t *option,
                      const char *ftitle,
                      Int_t compress,
                      Int_t netopt,
                      Bool_t parallelopen = kFALSE);

  // This matches everything else.
  TStorageFactoryFile(const char *name, Option_t *option = "", const char *ftitle = "", Int_t compress = 1);

  ~TStorageFactoryFile(void) override;

  Bool_t ReadBuffer(char *buf, Int_t len) override;
  Bool_t ReadBuffer(char *buf, Long64_t pos, Int_t len) override;
  Bool_t ReadBufferAsync(Long64_t off, Int_t len) override;
  Bool_t ReadBuffers(char *buf, Long64_t *pos, Int_t *len, Int_t nbuf) override;
  Bool_t WriteBuffer(const char *buf, Int_t len) override;

  void ResetErrno(void) const override;

protected:
  Int_t SysOpen(const char *pathname, Int_t flags, UInt_t mode) override;
  Int_t SysClose(Int_t fd) override;
  Long64_t SysSeek(Int_t fd, Long64_t offset, Int_t whence) override;
  Int_t SysStat(Int_t fd, Long_t *id, Long64_t *size, Long_t *flags, Long_t *modtime) override;
  Int_t SysSync(Int_t fd) override;

private:
  void Initialize(const char *name, Option_t *option = "");

  Bool_t ReadBuffersSync(char *buf, Long64_t *pos, Int_t *len, Int_t nbuf);

  void releaseStorage() { get_underlying_safe(storage_).release(); }

  TStorageFactoryFile(void);

  edm::propagate_const<std::unique_ptr<Storage>> storage_;  //< Real underlying storage
};

#endif  // TFILE_ADAPTOR_TSTORAGE_FACTORY_FILE_H
