#ifndef TFILE_ADAPTOR_TSTORAGE_FACTORY_SYSTEM_H
# define TFILE_ADAPTOR_TSTORAGE_FACTORY_SYSTEM_H

//<<<<<< INCLUDES                                                       >>>>>>

# include "TSystem.h"

//<<<<<< PUBLIC DEFINES                                                 >>>>>>
//<<<<<< PUBLIC CONSTANTS                                               >>>>>>
//<<<<<< PUBLIC TYPES                                                   >>>>>>

namespace seal { class Storage; }

//<<<<<< PUBLIC VARIABLES                                               >>>>>>
//<<<<<< PUBLIC FUNCTIONS                                               >>>>>>
//<<<<<< CLASS DECLARATIONS                                             >>>>>>

/** TSystem wrapper around #StorageFactory and SEAL's #Storage.
    This class is a blatant copy of TDCacheSystem.  */
class TStorageFactorySystem : public TSystem
{
private:
    void		*fDirp;		// Directory handler
    void *		GetDirPt (void) const { return fDirp; }

public:
    ClassDef (TStorageFactorySystem, 0); // ROOT System operating on SEAL's Storage.

    TStorageFactorySystem (void);
    ~TStorageFactorySystem (void);

    virtual Int_t	MakeDirectory (const char *name);
    virtual void *	OpenDirectory (const char *name);
    virtual void	FreeDirectory (void *dirp);
    virtual const char *GetDirEntry (void *dirp);

    virtual Int_t	GetPathInfo (const char *path, FileStat_t &info);

    virtual Bool_t	AccessPathName (const char *path, EAccessMode mode);
};

//<<<<<< INLINE PUBLIC FUNCTIONS                                        >>>>>>
//<<<<<< INLINE MEMBER FUNCTIONS                                        >>>>>>

#endif // TFILE_ADAPTOR_TSTORAGE_FACTORY_SYSTEM_H
