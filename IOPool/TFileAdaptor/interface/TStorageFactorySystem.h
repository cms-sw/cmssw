#ifndef TFILE_ADAPTOR_TSTORAGE_FACTORY_SYSTEM_H
# define TFILE_ADAPTOR_TSTORAGE_FACTORY_SYSTEM_H

# include "TSystem.h"

class Storage;

/** TSystem wrapper around #StorageFactory and CMS #Storage.
  This class is a blatant copy of TDCacheSystem.  */
class TStorageFactorySystem : public TSystem
{
private:
  void			*fDirp;	// Directory handle
  void *		GetDirPt(void) const { return fDirp; }

public:
  ClassDef(TStorageFactorySystem, 0); // ROOT System operating on CMS Storage.

  TStorageFactorySystem(const char *, Bool_t); // For compatibility with TXNetFile, we don't actually use the arguments
  TStorageFactorySystem(void);
  ~TStorageFactorySystem(void);

  virtual Int_t		MakeDirectory(const char *name);
  virtual void *	OpenDirectory(const char *name);
  virtual void		FreeDirectory(void *dirp);
  virtual const char *	GetDirEntry(void *dirp);

  virtual Int_t		GetPathInfo(const char *path, FileStat_t &info);

  virtual Bool_t	AccessPathName(const char *path, EAccessMode mode);
};

#endif // TFILE_ADAPTOR_TSTORAGE_FACTORY_SYSTEM_H
