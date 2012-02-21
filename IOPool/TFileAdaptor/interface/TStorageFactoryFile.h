#ifndef TFILE_ADAPTOR_TSTORAGE_FACTORY_FILE_H
# define TFILE_ADAPTOR_TSTORAGE_FACTORY_FILE_H

# include "TFile.h"

#define READ_COALESCE_SIZE 256 * 1024

class Storage;

/** TFile wrapper around #StorageFactory and #Storage.  */
class TStorageFactoryFile : public TFile
{
public:
  ClassDef(TStorageFactoryFile, 0); // ROOT File operating on CMS Storage.

  // Due to limitations in the ROOT plugin manager, TStorageFactoryFile must
  // provide a constructor matching all the different variants that other
  // ROOT plugins might use

  // This one is to match TXNetFile
  TStorageFactoryFile(const char *name, Option_t *option,
                      const char *ftitle, Int_t compress, Int_t netopt,
                      Bool_t parallelopen);

  // This matches everything else.
  TStorageFactoryFile(const char *name, Option_t *option = "",
                      const char *ftitle = "", Int_t compress = 1);

  ~TStorageFactoryFile(void);

  virtual Bool_t	ReadBuffer(char *buf, Int_t len);
  virtual Bool_t	ReadBuffer(char *buf, Long64_t pos, Int_t len);
  virtual Bool_t	ReadBufferAsync(Long64_t off, Int_t len);
  virtual Bool_t	ReadBuffers(char *buf,  Long64_t *pos, Int_t *len, Int_t nbuf);
  virtual Bool_t	WriteBuffer(const char *buf, Int_t len);

  void			ResetErrno(void) const;

protected:
  virtual Int_t		SysOpen(const char *pathname, Int_t flags, UInt_t mode);
  virtual Int_t		SysClose(Int_t fd);
  virtual Long64_t	SysSeek(Int_t fd, Long64_t offset, Int_t whence);
  virtual Int_t		SysStat(Int_t fd, Long_t *id, Long64_t *size, Long_t *flags, Long_t *modtime);
  virtual Int_t		SysSync(Int_t fd);

private:
  void                  Initialize(const char *name, Option_t *option = "");

  TStorageFactoryFile(void);

  Storage		*storage_;		//< Real underlying storage
};

#endif // TFILE_ADAPTOR_TSTORAGE_FACTORY_FILE_H
