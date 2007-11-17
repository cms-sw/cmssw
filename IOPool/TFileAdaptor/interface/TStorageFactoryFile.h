#ifndef TFILE_ADAPTOR_TSTORAGE_FACTORY_FILE_H
# define TFILE_ADAPTOR_TSTORAGE_FACTORY_FILE_H

# include "TFile.h"

class Storage;

/** TFile wrapper around #StorageFactory and #Storage.  */
class TStorageFactoryFile : public TFile
{
public:
  ClassDef (TStorageFactoryFile, 0); // ROOT File operating on CMS Storage.

  // Default parameters for caching.  We turn the cache off as there
  // is no good general default.  ROOT's TFile defaults (10 MB, 512 kB
  // page size) gives reasonable performance but consumes a lot of memory.
  enum { kDefaultCacheSize = 0 }; // Default cache size in megabytes
  enum { kDefaultPageSize = 0 };  // Default page size, must be power of two

  TStorageFactoryFile (const char *name, Option_t *option = "",
		    	 const char *ftitle = "", Int_t compress = 1);
  ~TStorageFactoryFile ();

  virtual Bool_t	ReadBuffer (char *buf, Int_t len);
  virtual Bool_t	WriteBuffer (const char *buf, Int_t len);

  void			ResetErrno (void) const;

  static void		DefaultBuffering (bool useit);
  static void		DefaultCaching (Int_t cacheSize = kDefaultCacheSize,
				        Int_t pageSize = kDefaultPageSize);

protected:
  virtual Int_t		SysOpen (const char *pathname, Int_t flags, UInt_t mode);
  virtual Int_t		SysClose (Int_t fd);
  virtual Int_t		SysRead (Int_t fd, void *buf, Int_t len);
  virtual Int_t		SysWrite (Int_t fd, const void *buf, Int_t len);
  virtual Long64_t	SysSeek (Int_t fd, Long64_t offset, Int_t whence);
  virtual Int_t		SysStat (Int_t fd, Long_t *id, Long64_t *size, Long_t *flags, Long_t *modtime);
  virtual Int_t		SysSync (Int_t fd);

private:
  TStorageFactoryFile ();
  Int_t			FlushLazySeek (void);

  Storage		*m_storage;		//< Real underlying storage
  Long64_t		m_offset;		//< My current file offset
  Int_t			m_recursiveRead;	//< Amount of recursive read, -1 = none
  Bool_t		m_lazySeek;		//< Indicates seek pending to run
  Long64_t		m_lazySeekOffset;	//< "offset" value for lazy seek

  static bool		s_bufferDefault;	//< Default buffering setting
  static Int_t		s_cacheDefaultCacheSize;//< Default cache size per file, in megabytes
  static Int_t		s_cacheDefaultPageSize;	//< Default cache page size
};

#endif // TFILE_ADAPTOR_TSTORAGE_FACTORY_FILE_H
