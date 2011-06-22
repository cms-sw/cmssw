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
      template<typename T>
      T * make() const {
         TDirectory *d = _cd();
         T* t = new T();
         ROOT::DirAutoAdd_t func = T::Class()->GetDirectoryAutoAdd();
         if (func) { TH1AddDirectorySentry sentry; func(t,d); } 
         else { d->Append(t); }
         return t;
      }
      /// make new ROOT object
      template<typename T, typename Arg1>
      T * make( const Arg1 & a1 ) const {
         TDirectory *d = _cd();
         T * t = new T( a1 );
         ROOT::DirAutoAdd_t func = T::Class()->GetDirectoryAutoAdd();
         if (func) { TH1AddDirectorySentry sentry; func(t,d); }
         else { d->Append(t); }
         return t; 
      }
      /// make new ROOT object
      template<typename T, typename Arg1, typename Arg2>
      T * make( const Arg1 & a1, const Arg2 & a2 ) const {
         TDirectory *d = _cd();
         T * t =  new T( a1, a2 );
         ROOT::DirAutoAdd_t func = T::Class()->GetDirectoryAutoAdd();
         if (func) { TH1AddDirectorySentry sentry; func(t,d); }
         else { d->Append(t); }
         return t;
      }
      /// make new ROOT object
      template<typename T, typename Arg1, typename Arg2, typename Arg3>
      T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3 ) const {
         TDirectory *d = _cd();
         T * t =  new T( a1, a2, a3 );
         ROOT::DirAutoAdd_t func = T::Class()->GetDirectoryAutoAdd();
         if (func) { TH1AddDirectorySentry sentry; func(t,d); }
         else { d->Append(t); }
         return t;
      }
      /// make new ROOT object
      template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4>
      T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4 ) const {
         TDirectory *d = _cd();
         T * t =  new T( a1, a2, a3, a4 );
         ROOT::DirAutoAdd_t func = T::Class()->GetDirectoryAutoAdd();
         if (func) { TH1AddDirectorySentry sentry; func(t,d); }
         else { d->Append(t); }
         return t;
      }
      /// make new ROOT object
      template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
               typename Arg5>
      T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
                const Arg5 & a5 ) const {
         TDirectory *d = _cd();
         T * t =  new T( a1, a2, a3, a4, a5 );
         ROOT::DirAutoAdd_t func = T::Class()->GetDirectoryAutoAdd();
         if (func) { TH1AddDirectorySentry sentry; func(t,d); }
         else { d->Append(t); }
         return t;
      }
      /// make new ROOT object
      template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
               typename Arg5, typename Arg6>
      T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
                const Arg5 & a5, const Arg6 & a6 ) const {
         TDirectory *d = _cd();
         T * t =  new T( a1, a2, a3, a4, a5, a6 );
         ROOT::DirAutoAdd_t func = T::Class()->GetDirectoryAutoAdd();
         if (func) { TH1AddDirectorySentry sentry; func(t,d); }
         else { d->Append(t); }
         return t;
      }
      /// make new ROOT object
      template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
               typename Arg5, typename Arg6, typename Arg7>
      T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
                const Arg5 & a5, const Arg6 & a6, const Arg7 & a7 ) const {
         TDirectory *d = _cd();
         T * t =  new T( a1, a2, a3, a4, a5, a6, a7 );
         ROOT::DirAutoAdd_t func = T::Class()->GetDirectoryAutoAdd();
         if (func) { TH1AddDirectorySentry sentry; func(t,d); }
         else { d->Append(t); }
         return t;
      }
      /// make new ROOT object
      template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
               typename Arg5, typename Arg6, typename Arg7, typename Arg8>
      T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
                const Arg5 & a5, const Arg6 & a6, const Arg7 & a7, const Arg8 & a8 ) const {
         TDirectory *d = _cd();
         T * t =  new T( a1, a2, a3, a4, a5, a6, a7, a8 );
         ROOT::DirAutoAdd_t func = T::Class()->GetDirectoryAutoAdd();
         if (func) { TH1AddDirectorySentry sentry; func(t,d); }
         else { d->Append(t); }
         return t;
      }
      /// make new ROOT object
      template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
               typename Arg5, typename Arg6, typename Arg7, typename Arg8, 
               typename Arg9>
      T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
                const Arg5 & a5, const Arg6 & a6, const Arg7 & a7, const Arg8 & a8,
                const Arg9 & a9 ) const {
         TDirectory *d = _cd();
         T * t =  new T( a1, a2, a3, a4, a5, a6, a7, a8, a9 );
         ROOT::DirAutoAdd_t func = T::Class()->GetDirectoryAutoAdd();
         if (func) { TH1AddDirectorySentry sentry; func(t,d); }
         else { d->Append(t); }
         return t;
      }
      /// make new ROOT object
      template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
               typename Arg5, typename Arg6, typename Arg7, typename Arg8, 
               typename Arg9, typename Arg10>
      T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
                const Arg5 & a5, const Arg6 & a6, const Arg7 & a7, const Arg8 & a8,
                const Arg9 & a9, const Arg10 & a10 ) const {
         TDirectory *d = _cd(); 
         T * t =  new T( a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 );
         ROOT::DirAutoAdd_t func = T::Class()->GetDirectoryAutoAdd();
         if (func) { TH1AddDirectorySentry sentry; func(t,d); }
         else { d->Append(t); }
         return t;
      }
      /// make new ROOT object
      template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
               typename Arg5, typename Arg6, typename Arg7, typename Arg8, 
               typename Arg9, typename Arg10, typename Arg11>
      T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
                const Arg5 & a5, const Arg6 & a6, const Arg7 & a7, const Arg8 & a8,
                const Arg9 & a9, const Arg10 & a10, const Arg11 & a11 ) const {
         TDirectory *d = _cd();
         T * t =  new T( a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 );
         ROOT::DirAutoAdd_t func = T::Class()->GetDirectoryAutoAdd();
         if (func) { TH1AddDirectorySentry sentry; func(t,d); }
         else { d->Append(t); }
         return t;
      }
      /// make new ROOT object
      template<typename T, typename Arg1, typename Arg2, typename Arg3, typename Arg4, 
               typename Arg5, typename Arg6, typename Arg7, typename Arg8, 
               typename Arg9, typename Arg10, typename Arg11, typename Arg12>
      T * make( const Arg1 & a1, const Arg2 & a2, const Arg3 & a3, const Arg4 & a4, 
                const Arg5 & a5, const Arg6 & a6, const Arg7 & a7, const Arg8 & a8,
                const Arg9 & a9, const Arg10 & a10, const Arg11 & a11, const Arg12 & a12 ) const {
         TDirectory *d = _cd();
         T * t =  new T( a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12 );
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
