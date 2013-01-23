// -*- C++ -*-
//
// Package:     LibraryLoader
// Class  :     RootAutoLibraryLoader
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:
//         Created:  Wed Nov 30 14:55:01 EST 2005
//

// system include files
#include <string>
#include <iostream>
#include <map>
#include "TROOT.h"
#include "TInterpreter.h"
#include "G__ci.h"

// user include files
#include "FWCore/RootAutoLibraryLoader/interface/RootAutoLibraryLoader.h"
#include "FWCore/RootAutoLibraryLoader/src/stdNamespaceAdder.h"

#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/ProblemTracker.h"
#include "FWCore/PluginManager/interface/PluginCapabilities.h"
#include "FWCore/Utilities/interface/DictionaryTools.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "Cintex/Cintex.h"
#include "TClass.h"

// We cannot use the MessageLogger here because this is also used by standalones that do not have the logger.

//
// constants, enums and typedefs
//
namespace {
  //Based on http://root.cern.ch/lxr/source/cintex/src/CINTSourceFile.h
  // If a Cint dictionary is accidently loaded as a side effect of loading a CMS
  // library Cint must have a file name assigned to that dictionary else Cint may crash
  class RootLoadFileSentry {
  public:
    RootLoadFileSentry() {
       G__setfilecontext("{CMS auto library loader}", &oldIFile_);
    }

    ~RootLoadFileSentry() {
      G__input_file* ifile = G__get_ifile();
      if (ifile) {
        *ifile = oldIFile_;
      }
    }

  private:
      G__input_file oldIFile_;
  };
}

//
// static data member definitions
//
//hold onto the previous autolibrary loader
typedef int (*CallbackPtr)(char*, char*);
static CallbackPtr gPrevious = nullptr;
static char const* kDummyLibName = "*dummy";

//This is actually defined within ROOT's v6_struct.cxx file but is not declared static
// I want to use it so that if the autoloading is already turned on, I can call the previously declared routine
extern CallbackPtr G__p_class_autoloading;

namespace ROOT {
  namespace Cintex {
    std::string CintName(std::string const&);
  }
}

namespace edm {
   namespace {

      std::map<std::string, std::string>&
      cintToReflexSpecialCasesMap() {
         static std::map<std::string, std::string> s_map;
         return s_map;
      }

      void
      addWrapperOfVectorOfBuiltin(std::map<std::string, std::string>& iMap, char const* iBuiltin) {
         static std::string sReflexPrefix("edm::Wrapper<std::vector<");
         static std::string sReflexPostfix("> >");

         //Wrapper<vector<float, allocator<float> > >
         static std::string sCintPrefix("Wrapper<vector<");
         static std::string sCintMiddle(",allocator<");
         static std::string sCintPostfix("> > >");

         std::string type(iBuiltin);
         iMap.insert(make_pair(sCintPrefix + type + sCintMiddle + type + sCintPostfix,
                               sReflexPrefix + type + sReflexPostfix));
      }

      std::string
      classNameForRoot(std::string const& classname) {
        // Converts the name to the name known by CINT (e.g. strips out "std::")
        return ROOT::Cintex::CintName(classname);
      }

      bool
      isDictionaryLoaded(std::string const& rootclassname) {
        // This checks if the class name is known to the interpreter.
        // In this context, this will be true if and only if the dictionary has been loaded (and converted to CINT).
        // This code is independent of the identity of the interpreter.
        ClassInfo_t* info = gInterpreter->ClassInfo_Factory(rootclassname.c_str());
        return gInterpreter->ClassInfo_IsValid(info);
      }

      bool loadLibraryForClass(char const* classname) {
        //std::cout << "loadLibaryForClass" << std::endl;
        if(nullptr == classname) {
          return false;
        }
        //std::cout << "asking to find " << classname << std::endl;
        std::string const& cPrefix = dictionaryPlugInPrefix();
        //std::cout << "asking to find " << cPrefix + classname << std::endl;
        std::string rootclassname = classNameForRoot(classname);
        try {
          //give ROOT a name for the file we are loading
          RootLoadFileSentry sentry;
          if(edmplugin::PluginCapabilities::get()->tryToLoad(cPrefix + classname)) {
            if(!isDictionaryLoaded(rootclassname)) {
              //would be nice to issue a warning here.  Not sure the remainder of this comment is correct.
              // this message happens too often (too many false positives) to be useful plus ROOT will complain about a missing dictionary
              //std::cerr << "Warning: ROOT knows about type '" << classname << "' but has no dictionary for it." << std::endl;
              return false;
            }
          } else {
            //see if adding a std namespace helps
            std::string name = root::stdNamespaceAdder(classname);
            //std::cout << "see if std helps" << std::endl;
            if (not edmplugin::PluginCapabilities::get()->tryToLoad(cPrefix + name)) {
              // Too many false positives on built-in types here.
              return false;
            }
            if(!isDictionaryLoaded(rootclassname)) {
              //would be nice to issue a warning here
              return false;
            }
          }
        } catch(cms::Exception& e) {
          //would be nice to issue a warning here
          return false;
        }
        //std::cout << "loaded " << classname << std::endl;
        return true;
      }

      //Based on code in ROOT's TCint.cxx file

      int ALL_AutoLoadCallback(char* c, char* l) {
        //NOTE: if the library (i.e. 'l') is an empty string this means we are dealing with a namespace
        // These checks appear to avoid a crash of ROOT during shutdown of the application
        if(nullptr == c || nullptr == l || l[0] == 0) {
          return 0;
        }
        ULong_t varp = G__getgvp();
        G__setgvp((long)G__PVOID);
        int result = loadLibraryForClass(c) ? 1:0;
        G__setgvp(varp);
        //NOTE: the check for the library is done since we can have a failure
        // if a CMS library has an incomplete set of reflex dictionaries where
        // the remaining dictionaries can be found by Cint.  If the library with
        // the reflex dictionaries is loaded first, then the Cint library then any
        // requests for a Type from the reflex library will fail because for
        // some reason the loading of the Cint library causes reflex to forget about
        // what types it already loaded from the reflex library.  This problem was
        // seen for libDataFormatsMath and libMathCore.  I do not print an error message
        // since the dictionaries are actually loaded so things work fine.
        if(!result && 0 != strcmp(l, kDummyLibName) && gPrevious) {
          result = gPrevious(c, l);
        }
        return result;
      }

      //Cint requires that we register the type and library containing the type
      // before the autoloading will work
        struct CompareFirst {
          bool operator()(std::pair<std::string, std::string> const& iLHS,
                          std::pair<std::string, std::string> const& iRHS) const{
            return iLHS.first > iRHS.first;
          }
        };

      void registerTypes() {
        edmplugin::PluginManager* db =  edmplugin::PluginManager::get();

        typedef edmplugin::PluginManager::CategoryToInfos CatToInfos;

        CatToInfos::const_iterator itFound = db->categoryToInfos().find("Capability");

        if(itFound == db->categoryToInfos().end()) {
          return;
        }

        //in order to determine if a name is from a class or a namespace, we will order
        // all the classes in descending order so that embedded classes will be seen before
        // their containing classes, that way we can say the containing class is a namespace
        // before finding out it is actually a class
        typedef std::vector<std::pair<std::string, std::string> > ClassAndLibraries;
        ClassAndLibraries classes;
        classes.reserve(1000);
        std::string lastClass;

        //find where special cases come from
        std::map<std::string, std::string> specialsToLib;
        std::map<std::string, std::string> const& specials = cintToReflexSpecialCasesMap();
        for(auto const& special : specials) {
          specialsToLib[classNameForRoot(special.second)];
        }
        std::string const& cPrefix = dictionaryPlugInPrefix();
        for(auto const& info : itFound->second) {
          if (lastClass == info.name_) {
            continue;
          }
          lastClass = info.name_;
          if(cPrefix == lastClass.substr(0, cPrefix.size())) {
            std::string className = classNameForRoot(lastClass.c_str() + cPrefix.size());
            classes.emplace_back(className, info.loadable_.string());
            std::map<std::string, std::string>::iterator iFound = specialsToLib.find(className);
            if(iFound != specialsToLib.end()) {
              // std::cout << "Found " << lastClass << " : " << className << std::endl;
              iFound->second = info.loadable_.string();
            }
          }
        }
        //sort_all(classes, std::greater<std::string>());
        //sort_all(classes, CompareFirst());
        //the values are already sorted by less, so just need to reverse to get greater
        for(ClassAndLibraries::reverse_iterator itClass = classes.rbegin(), itClassEnd = classes.rend();
            itClass != itClassEnd;
            ++itClass) {

          std::string const& className = itClass->first;
          std::string const& libraryName = itClass->second;
          //need to register namespaces and figure out if we have an embedded class
          static std::string const toFind(":<");
          std::string::size_type pos = 0;
          while(std::string::npos != (pos = className.find_first_of(toFind, pos))) {
            if (className[pos] == '<') {break;}
            if (className.size() <= pos + 1 || className[pos + 1] != ':') {break;}
            //should check to see if this is a class or not
            G__set_class_autoloading_table(const_cast<char*>(className.substr(0, pos).c_str()), const_cast<char*>(""));
            //std::cout << "namespace " << className.substr(0, pos).c_str() << std::endl;
            pos += 2;
          }
          G__set_class_autoloading_table(const_cast<char*>(className.c_str()), const_cast<char*>(libraryName.c_str()));
          //std::cout << "class " << className.c_str() << std::endl;
        }

        //now handle the special cases
        for(auto const& special : specials) {
          std::string const& name = special.second;
          std::string rootname = classNameForRoot(name);
          //std::cout << "registering special " << itSpecial->first << " " << name << " " << std::endl << "    " << specialsToLib[rootname] << std::endl;
          //force loading of specials
          if(specialsToLib[rootname].size()) {
            //std::cout << "&&&&& found special case " << itSpecial->first << std::endl;
            if(!isDictionaryLoaded(rootname) and
                (not edmplugin::PluginCapabilities::get()->tryToLoad(cPrefix + name))) {
              std::cout << "failed to load plugin for " << cPrefix + name << std::endl;
              continue;
            } else {
              //need to construct the Class ourselves
              if(!isDictionaryLoaded(rootname)) {
                std::cout << "dictionary did not build " << name << std::endl;
                continue;
              }
              TClass* namedClass = TClass::GetClass(rootname.c_str());
              if(nullptr == namedClass) {
                std::cout << "failed to get TClass for " << name << std::endl;
                continue;
              }
              namedClass->Clone(special.first.c_str());
              std::string magictypedef("namespace edm { typedef ");
              magictypedef += rootname + " " + special.first + "; }";
              // std::cout << "Magic typedef " << magictypedef << std::endl;
              gROOT->ProcessLine(magictypedef.c_str());
            }
          }
        }
      }
   }

   //
   // constructors and destructor
   //
   RootAutoLibraryLoader::RootAutoLibraryLoader() :
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,27,6)
     classNameAttemptingToLoad_(nullptr),
     isInitializingCintex_(true) {
#else
     classNameAttemptingToLoad_(nullptr) {
#endif
      AssertHandler h;
      gROOT->AddClassGenerator(this);
      ROOT::Cintex::Cintex::Enable();
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,27,6)
      isInitializingCintex_ =false;
#endif
      //set the special cases
      std::map<std::string, std::string>& specials = cintToReflexSpecialCasesMap();
      if(specials.empty()) {
         addWrapperOfVectorOfBuiltin(specials,"bool");

         addWrapperOfVectorOfBuiltin(specials,"char");
         addWrapperOfVectorOfBuiltin(specials,"unsigned char");
         addWrapperOfVectorOfBuiltin(specials,"signed char");
         addWrapperOfVectorOfBuiltin(specials,"short");
         addWrapperOfVectorOfBuiltin(specials,"unsigned short");
         addWrapperOfVectorOfBuiltin(specials,"int");
         addWrapperOfVectorOfBuiltin(specials,"unsigned int");
         addWrapperOfVectorOfBuiltin(specials,"long");
         addWrapperOfVectorOfBuiltin(specials,"unsigned long");
         addWrapperOfVectorOfBuiltin(specials,"long long");
         addWrapperOfVectorOfBuiltin(specials,"unsigned long long");

         addWrapperOfVectorOfBuiltin(specials,"float");
         addWrapperOfVectorOfBuiltin(specials,"double");
      }
      //std::cout << "my loader" << std::endl;
      //remember if the callback was already set so we can chain together our results
      gPrevious = G__p_class_autoloading;
      G__set_class_autoloading_callback(&ALL_AutoLoadCallback);
      registerTypes();
   }

   //
   // member functions
   //

   TClass*
   RootAutoLibraryLoader::GetClass(char const* classname, Bool_t load) {
      TClass* returnValue = nullptr;
      if(classNameAttemptingToLoad_ != nullptr && !strcmp(classname, classNameAttemptingToLoad_)) {
         // We can try to see if the class name contains "basic_string<char>".
         // If so, we replace "basic_string<char>" with "string" and try again.
         std::string className(classname);
         std::string::size_type idx = className.find("basic_string<char>");
         if (idx != std::string::npos) {
            className.replace(idx, 18, std::string("string"));
            //if basic_string<char> was the last argument to a templated class
            // then there would be an extra space to separate the two '>'
            if(className.size() > idx + 6 && className[idx + 6] == ' ') {
              className.replace(idx + 6, 1, "");
            }
            classNameAttemptingToLoad_ = className.c_str();
            returnValue = gROOT->GetClass(className.c_str(), kTRUE);
            classNameAttemptingToLoad_ = classname;
            return returnValue;
         }
         //NOTE: As of ROOT 5.27.06 this warning generates false positives for HepMC classes because
         // ROOT has special handling for them built into class.rules
         //std::cerr << "WARNING[RootAutoLibraryLoader]: ROOT failed to create CINT dictionary for " << classname << std::endl;
         return nullptr;
      }
      //std::cout << "looking for " << classname << " load " << (load? "T":"F") << std::endl;
      if (load) {
        //std::cout << " going to call loadLibraryForClass" << std::endl;
        //[ROOT 5.28] When Cintex is in its 'Enable' method it will register callbacks to build
        // TClasses. During this phase we do not want to actually force TClasses to have to 
        // come into existence.
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,27,6)
        if (not isInitializingCintex_ and loadLibraryForClass(classname)) {
#else
         if (loadLibraryForClass(classname)) {
#endif
          //use this to check for infinite recursion attempt
          classNameAttemptingToLoad_ = classname;
          // This next call will create the TClass object for the class.
          // It will also attempt to load the dictionary for the class
          // if the second argument is kTRUE. This is the default, so it
          // need not be explicitly specified.
          returnValue = gROOT->GetClass(classname, kTRUE);
          classNameAttemptingToLoad_ = nullptr;
        }
      }
      return returnValue;
   }

   TClass*
   RootAutoLibraryLoader::GetClass(type_info const& typeinfo, Bool_t load) {
     //std::cout << "looking for type " << typeinfo.name() << std::endl;
      TClass* returnValue = nullptr;
      if(load) {
         return GetClass(typeinfo.name(), load);
      }
      return returnValue;
   }

   void
   RootAutoLibraryLoader::enable() {
      //static BareRootProductGetter s_getter;
      //static EDProductGetter::Operate s_op(&s_getter);
      static RootAutoLibraryLoader s_loader;
   }

   void
   RootAutoLibraryLoader::loadAll() {
     // std::cout << "LoadAllDictionaries" << std::endl;
     enable();

     edmplugin::PluginManager* db =  edmplugin::PluginManager::get();

     typedef edmplugin::PluginManager::CategoryToInfos CatToInfos;

     CatToInfos::const_iterator itFound = db->categoryToInfos().find("Capability");

     if(itFound == db->categoryToInfos().end()) {
       return;
     }
     std::string lastClass;

     //give ROOT a name for the file we are loading
     RootLoadFileSentry sentry;

     for(auto const& info : itFound->second) {
       if (lastClass == info.name_) {
         continue;
       }

       lastClass = info.name_;
       edmplugin::PluginCapabilities::get()->load(lastClass);
       //NOTE: since we have the library already, we could be more efficient if we just load it ourselves
     }
   }
}

//ClassImp(RootAutoLibraryLoader)
