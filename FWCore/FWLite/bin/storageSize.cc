// -*- C++ -*-
//
// Package:     FWLite
// Class  :     storageSize
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Thu Jan 20 09:50:58 CST 2011
//

// system include files
#include <iostream>
#include <boost/program_options.hpp>
#include "TClass.h"
#include "TBufferFile.h"

// user include files
#include "FWCore/FWLite/interface/AutoLibraryLoader.h"
#include "FWCore/Utilities/interface/Exception.h"


//
// constants, enums and typedefs
//
static char const* const kClassNameOpt = "className";
static char const* const kHelpOpt = "help";
static char const* const kHelpCommandOpt="help,h";


int main(int argc, char* argv[]) try
{
   std::string descString(argv[0]);
   descString += " [options] [--";
   descString += kClassNameOpt;
   descString += "] class_name"
   "\n The program dumps information about how much storage space is needed to store the class"
   "\nAllowed options";
   boost::program_options::options_description desc(descString);
   desc.add_options()
   (kHelpCommandOpt, "show this help message")
   (kClassNameOpt,"name of class");

   boost::program_options::positional_options_description p;
   p.add(kClassNameOpt, 1);

   boost::program_options::variables_map vm;
   try {
      store(boost::program_options::command_line_parser(argc,argv).options(desc).positional(p).run(),vm);
      notify(vm);
   } catch(boost::program_options::error const& iException) {
      std::cerr <<"failed to parse command line \n"<<iException.what()<<"\n";
      return 1;
   }
   
   if(vm.count(kHelpOpt)) {
      std::cout << desc <<std::endl;
      return 0;
   }
   
   if(!vm.count(kClassNameOpt)) {
      std::cerr <<"no class name given\n";
      return 1;
   }
   
   std::string className(vm[kClassNameOpt].as<std::string>());
   
   AutoLibraryLoader::enable();

   TClass* cls = TClass::GetClass(className.c_str());
   if(0==cls) {
      std::cerr <<"class '"<<className<<"' is unknown by ROOT\n";
      return 1;
   }
   
   void* objInstance = cls->New();
   if(0==objInstance) {
      std::cerr <<"unable to create a default instance of the class "<<className;
      return 1;
   }
   
   TBufferFile bf(TBuffer::kWrite);
   
   gDebug = 3;
   cls->WriteBuffer(bf, objInstance);
 
   gDebug = 0;
   std::cout <<"Total amount stored: "<<bf.Length()<<" bytes"<<std::endl;
   std::cout<<"\nNOTE: add 4 bytes for each 'has written' value because of a bug in ROOT's printout of the accounting"
   <<"\n  Each class (inheriting or as member data) has metadata an overhead of 10 bytes"<<std::endl;
   return 0;
} catch(cms::Exception const& e) {
  std::cerr << e.explainSelf() << std::endl;
  return 1;
} catch(std::exception const& e) {
  std::cerr << e.what() << std::endl;
  return 1;
}
