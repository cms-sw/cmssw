#ifndef FWCore_Services_PrintLoadingPlugins_h
#define FWCore_Services_PrintLoadingPlugins_h

#include "FWCore/Services/interface/PrintLoadingPlugins.h"
#include <boost/filesystem/path.hpp>

// -*- C++ -*-
//
// Package:     Services
// Class  :     PrintLoadingPlugins
// 
/**\class PrintLoadingPlugins PrintLoadingPlugins.h FWCore/Services/interface/PrintLoadingPlugins.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  
//         Created:  Thu Dec 13 11:17:02 EST 2007
// $Id: PrintLoadingPlugins.h,v 1.1 2008/01/14 18:14:24 chrjones Exp $
//

// system include files

// user include files

// forward declarations

class PrintLoadingPlugins
{

   public:
      PrintLoadingPlugins();
      virtual ~PrintLoadingPlugins();

      void goingToLoad(const boost::filesystem::path&);

      void askedToLoad(const std::string& ,const std::string& );

       // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------

   private:
       
      PrintLoadingPlugins(const PrintLoadingPlugins&); // stop default

      const PrintLoadingPlugins& operator=(const PrintLoadingPlugins&); // stop default

      // ---------- member data --------------------------------

};


#endif
