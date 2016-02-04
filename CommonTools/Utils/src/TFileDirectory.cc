#include <ostream>
#include <iostream>

#include "CommonTools/Utils/interface/TFileDirectory.h"

#include "boost/regex.hpp"



using namespace std;

TDirectory * 
TFileDirectory::getBareDirectory (const string &subdir) const 
{
   return _cd (subdir, false);
}

bool
TFileDirectory::cd () const
{
   _cd ("", false);
   return true;
}

TDirectory *
TFileDirectory::_cd (const string &subdir, bool createNeededDirectories) const
{
   string fpath = fullPath();
   if (subdir.length())
   {
      // not empty, we need to append it to path
      if (fpath.length())
      {
         // path is also not empty, so add a slash and let's get going.
         fpath += "/" + subdir;
      } else {
         // path doesn't exist, so just use subdir
         fpath = subdir;
      }
   }
   TDirectory * dir = file_->GetDirectory( fpath.c_str() );
   if ( dir == 0 ) 
   {      
      // we didn't find the directory
      //
      // If we're not supposed to create the diretory, then we should
      // complain now that it doesn't exist.
      if (! createNeededDirectories)
      {
         cout << "here " << fpath << endl;
         throw 
            cms::Exception( "InvalidDirectory" ) 
            << "directory " << fpath << " doesn't exist.";
      }
      if ( ! path_.empty() ) 
      {
         dir = file_->GetDirectory( path_.c_str() );
         if ( dir == 0 )
         {
            throw 
               cms::Exception( "InvalidDirectory" ) 
               << "Can't change directory to path: " << path_;
         }
      } else 
      {
         // the base path 'path_' is empty, so just use the pointer to
         // the file.
         dir = file_;
      }
      // if a subdirectory was passed in, then this directory better
      // already exist (since you shoudln't be cd'ing into a directory
      // before making it and the cd with a subdir is only used to get
      // histograms that are already made).
      if (subdir.length())
      {
         throw 
            cms::Exception( "InvalidDirectory" ) 
            << "directory " << fpath << " doesn't exist.";
      }
      // if we're here, then that means that this is the first time
      // that we are calling cd() on this directory AND cd() has not
      // been called with a subdirectory, so go ahead and make the
      // directory.
      dir = _mkdir (dir, dir_, descr_);
   }
   bool ok = file_->cd( fpath.c_str() );
   if ( ! ok )
   {
      throw 
         cms::Exception( "InvalidDirectory" ) 
         << "Can't change directory to path: " << fpath;
   }
   return dir;
}

TDirectory*
TFileDirectory::_mkdir (TDirectory *dirPtr, 
                        const string &subdirName, 
                        const string &description) const
{
   // do we have this one already
   TDirectory *subdirPtr = dirPtr->GetDirectory (subdirName.c_str());
   if (subdirPtr)
   {
      subdirPtr->cd();
      return subdirPtr;
   }
   // if we're here, then this directory doesn't exist.  Is this
   // directory a subdirectory?
   const boost::regex subdirRE ("(.+?)/([^/]+)");
   boost::smatch matches;
   TDirectory *parentDir = 0;
   string useName = subdirName;
   if( boost::regex_match (subdirName, matches, subdirRE) )
   {
      parentDir = _mkdir (dirPtr, matches[1], description);
      useName = matches[2];
   } else {
      // This is not a subdirectory, so we're golden
      parentDir = dirPtr;
   }
   subdirPtr = parentDir->mkdir (useName.c_str());
   if ( ! subdirPtr )
   {
      throw 
         cms::Exception( "InvalidDirectory" ) 
            << "Can't create directory " << dir_ << " in path: " << path_;
   }
   subdirPtr->cd();
   return subdirPtr;
}

TObject*
TFileDirectory::_getObj (const string &objname, const string &subdir) const
{
   TObject *objPtr = getBareDirectory (subdir)->Get (objname.c_str());
   if ( ! objPtr)
   {
      // no histogram found by that name.  Sorry dude.
      if (subdir.length())
      {
         throw
            cms::Exception ("ObjectNotFound")
            << "Can not find object named " << objname
            << " in subdir " << subdir;
      } else {
         throw
            cms::Exception ("ObjectNotFound")
            << "Can not find object named " << objname;
      }
   } // if nothing found
   return objPtr;
}

std::string 
TFileDirectory::fullPath() const 
{
   return string( path_.empty() ? dir_ : path_ + "/" + dir_ );
}

TFileDirectory 
TFileDirectory::mkdir( const std::string & dir, const std::string & descr ) 
{
   TH1AddDirectorySentry sentry;
   _cd();
   return TFileDirectory( dir, descr, file_, fullPath() );
}
