#ifndef FWCore_PluginManager_CacheParser_h
#define FWCore_PluginManager_CacheParser_h
// -*- C++ -*-
//
// Package:     PluginManager
// Class  :     CacheParser
// 
/**\class CacheParser CacheParser.h FWCore/PluginManager/interface/CacheParser.h

 Description: Parses the cache information about which plugins are in which libraries

 Usage:
    The format expected is line oriented, with each lying containing

    <file name> <plugin name> <plugin type>

If a space exists in either of these three fields, it will be replaced with a %, E.g.
    /local/lib/pluginFoo.so Widget Fabulous%Bar%Stuff

*/
//
// Original Author:  Chris Jones
//         Created:  Wed Apr  4 14:30:51 EDT 2007
// $Id: CacheParser.h,v 1.2 2007/04/12 12:51:11 wmtan Exp $
//

// system include files
#include <iosfwd>
#include <string>
#include <map>
#include <vector>
#include <boost/filesystem/path.hpp>

// user include files
#include "FWCore/PluginManager/interface/PluginInfo.h"

// forward declarations
namespace edmplugin {
class CacheParser
{

   public:
      typedef std::map<std::string, std::vector<PluginInfo> > CategoryToInfos;
      typedef std::pair< std::string, std::string> NameAndType;
      typedef std::vector< NameAndType > NameAndTypes;
      typedef std::map<boost::filesystem::path, NameAndTypes > LoadableToPlugins;

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
      /**The std::vector<PluginInfo>'s in CategoryToInfos are guaranteed to be ordered by
        PluginInfo.name_ where identical names are ordered by the order they are passed to read.
        In this way multiple calls to read for different directories will preserve the ordering
        */
      static void read(std::istream&, const boost::filesystem::path& iDirectory, CategoryToInfos& oOut);
      static void write(const CategoryToInfos&, std::ostream&);
      
      static void read(std::istream&, LoadableToPlugins& oOut);
      static void write(LoadableToPlugins& iIn, std::ostream&);
   private:
      CacheParser(const CacheParser&); // stop default

      const CacheParser& operator=(const CacheParser&); // stop default

      static bool readline(std::istream& iIn, const boost::filesystem::path& iDirectory,
               unsigned long iRecordNumber, PluginInfo &oInfo, std::string& oPluginType);
      static std::string& replaceSpaces(std::string& io);
      static std::string& restoreSpaces(std::string& io);
      
      // ---------- member data --------------------------------

};

}
#endif
