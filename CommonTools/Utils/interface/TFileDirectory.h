#ifndef Utils_TFileDirectory_h
#define Utils_TFileDirectory_h
/* \class TFileDirectory
 *
 * \author Luca Lista, INFN
 *
 */
#include <string>
#include "FWCore/Utilities/interface/Exception.h"
#include "CommonTools/Utils/interface/TH1AddDirectorySentry.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TClass.h"
#include "TH1.h"

namespace fwlite {
   class TFileService;
}

class TFileService;
class TFile;
class TDirectory; 

class TFileDirectory {
   public:

      TFileDirectory() : file_(nullptr), dir_(), descr_(), path_() {
      }

      /// descructor
      virtual ~TFileDirectory() { }

      // cd()s to requested directory and returns true (if it is not
      // able to cd, it throws exception).
      bool cd () const;

      // returns a TDirectory pointer
      TDirectory *getBareDirectory (const std::string &subdir = "") const;
      
      // reutrns a "T" pointer matched to objname
      template< typename T > T* getObject (const std::string &objname,
                                           const std::string &subdir = "")
      {
         TObject *objPtr = _getObj (objname, subdir);
         // Ok, we've got it.  Let's see if it's a histogram
         T * retval = dynamic_cast< T* > ( objPtr );
         if ( ! retval )
         {
            // object isn't a of class T
            throw
               cms::Exception ("ObjectNotCorrectlyTyped")
               << "Object named " << objname << " is not of correct type";
         }
         return retval;
      }

      /// make new ROOT object
      template<typename T, typename ... Args>
      T* make(const Args& ... args) const {
         TDirectory *d = _cd();
         T* t = new T(args ...);
         ROOT::DirAutoAdd_t func = T::Class()->GetDirectoryAutoAdd();
         if (func) { TH1AddDirectorySentry sentry; func(t,d); }
         else { d->Append(t); }
         return t;
      }

      /// create a new subdirectory
      TFileDirectory mkdir( const std::string & dir, const std::string & descr = "" );
      /// return the full path of the stored histograms
      std::string fullPath() const;

   private:
      
      TObject* _getObj (const std::string &objname, 
                        const std::string &subdir = "") const;

      TDirectory* _cd (const std::string &subdir = "",
                       bool createNeededDirectories = true) const;

      // it's completely insane that this needs to be const since
      // 'mkdir' clearly changes things, but that's the way the cookie
      // crumbles...
      TDirectory* _mkdir (TDirectory *dirPtr,
                          const std::string &dir, 
                          const std::string &description) const;

      TFileDirectory( const std::string & dir, const std::string & descr,
                      TFile * file, const std::string & path ) : 
         file_( file ), dir_( dir ), descr_( descr ), path_( path ) {
      }
      friend class TFileService;
      friend class fwlite::TFileService;
      TFile * file_;
      std::string dir_, descr_, path_;
};

#endif
