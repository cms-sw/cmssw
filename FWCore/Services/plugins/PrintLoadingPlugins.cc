// -*- C++ -*-
//
// Package:     Services
// Class  :     PrintLoadingPlugins
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Thu Dec 13 15:00:49 EST 2007
//

// system include files

// user include files
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/PluginInfo.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "FWCore/Utilities/interface/Signal.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <string>
#include <map>

class PrintLoadingPlugins
{
  
public:
  PrintLoadingPlugins();
  virtual ~PrintLoadingPlugins();
  
  void goingToLoad(const boost::filesystem::path&);
  
  void askedToLoad(const std::string& ,const std::string& );
  
  // ---------- const member functions ---------------------
  
  // ---------- static member functions --------------------
  static void fillDescriptions(edm::ConfigurationDescriptions & descriptions);
  
  // ---------- member functions ---------------------------
  
private:
  
  PrintLoadingPlugins(const PrintLoadingPlugins&); // stop default
  
  const PrintLoadingPlugins& operator=(const PrintLoadingPlugins&); // stop default
  
  // ---------- member data --------------------------------
  
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
using namespace edmplugin;

PrintLoadingPlugins::PrintLoadingPlugins()
{   
   using std::placeholders::_1;
   using std::placeholders::_2;
   PluginManager *pm = PluginManager::get();

   pm->askedToLoadCategoryWithPlugin_.connect(std::bind(std::mem_fn(&PrintLoadingPlugins::askedToLoad),this, _1,_2));
   
   pm->goingToLoad_.connect(std::bind(std::mem_fn(&PrintLoadingPlugins::goingToLoad),this, _1));

   
  
}

// PrintLoadingPlugins::PrintLoadingPlugins(const PrintLoadingPlugins& rhs)
// {
//    // do actual copying here;
// }

PrintLoadingPlugins::~PrintLoadingPlugins()
{
}

void PrintLoadingPlugins::fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.add("PrintLoadingPlugins", desc);
  descriptions.setComment("This service logs each request to load a plugin.");
}

//
// assignment operators
//
// const PrintLoadingPlugins& PrintLoadingPlugins::operator=(const PrintLoadingPlugins& rhs)
// {
//   //An exception safe implementation is
//   PrintLoadingPlugins temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

 namespace{
    struct PICompare {
	  bool operator()(const PluginInfo& iLHS,
			  const PluginInfo& iRHS) const {
	     return iLHS.name_ < iRHS.name_;
	  }
    };
 }

void PrintLoadingPlugins::askedToLoad(const std::string& iCategory,
				      const std::string& iPlugin)
{ 
   PluginManager *pm = PluginManager::get();

   const PluginManager::CategoryToInfos& category = pm->categoryToInfos();

   PluginManager::CategoryToInfos::const_iterator itFound = category.find(iCategory);

   std::string libname("Not found");
   
   if(itFound != category.end()) {
      
      PluginInfo i;
      
      i.name_ = iPlugin;
      
      typedef std::vector<PluginInfo>::const_iterator PIItr;
      
      std::pair<PIItr,PIItr> range = std::equal_range(itFound->second.begin(),itFound->second.end(),i,PICompare());
      
      if(range.second - range.first > 1){
	 
	 const boost::filesystem::path& loadable = range.first->loadable_;
	 
	 libname = loadable.string();
	 
      }
      
      edm::LogAbsolute("GetPlugin")<<"Getting> '"<<iCategory<< "' "<<iPlugin 
				   <<"\n         from "<<libname <<std::endl;
   }
   
}

void PrintLoadingPlugins::goingToLoad(const boost::filesystem::path& Loadable_)

{
  edm::LogAbsolute("LoadLib")<<"Loading> "<<Loadable_.string()<< std::endl;
}


//
// const member functions
//

//
// static member functions
//

typedef edm::serviceregistry::NoArgsMaker<PrintLoadingPlugins> PrintLoadingPluginsMaker;
DEFINE_FWK_SERVICE_MAKER(PrintLoadingPlugins, PrintLoadingPluginsMaker);

