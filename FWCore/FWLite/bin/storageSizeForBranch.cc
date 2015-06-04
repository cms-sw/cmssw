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
#include "TFile.h"
#include "TBranch.h"
#include "TBufferFile.h"
#include "TTree.h"

// user include files
#include "FWCore/FWLite/interface/FWLiteEnabler.h"
#include "FWCore/Utilities/interface/Exception.h"


//
// constants, enums and typedefs
//
static char const* const kBranchNameOpt = "branchName";
static char const* const kFileNameOpt = "fileName";
static char const* const kHelpOpt = "help";
static char const* const kHelpCommandOpt="help,h";
static char const* const kEntryNumberCommandOpt = "entryNumber,e";
static char const* const kEntryNumberOpt = "entryNumber";

int main(int argc, char* argv[]) try
{
   std::string descString(argv[0]);
   descString += " [options] [--";
   descString += kBranchNameOpt;
   descString += "] branch_name [--";
   descString += kFileNameOpt;
   descString += "] file_name"
   "\n The program dumps information about how much storage space is needed to store a given TBranch within a ROOT file"
   "\nAllowed options";
   boost::program_options::options_description desc(descString);
   desc.add_options()
   (kHelpCommandOpt, "show this help message")
   (kEntryNumberCommandOpt, boost::program_options::value<int>(),"read branch from the given entry value (default 0)")
   (kBranchNameOpt,"name of branch")
   (kFileNameOpt,"name of file");

   boost::program_options::positional_options_description p;
   p.add(kBranchNameOpt, 1);
   p.add(kFileNameOpt, 1);

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
   int entryNumber = 0;
   if(vm.count(kEntryNumberOpt)) {
      entryNumber = vm[kEntryNumberOpt].as<int>();
   }
   
   if(!vm.count(kBranchNameOpt)) {
      std::cerr <<"no branch name given\n";
      return 1;
   }

   
   if(!vm.count(kFileNameOpt)) {
      std::cerr <<"no branch name given\n";
      return 1;
   }
   
   std::string branchName(vm[kBranchNameOpt].as<std::string>());
   std::string fileName(vm[kFileNameOpt].as<std::string>());
   
   TFile* file = TFile::Open(fileName.c_str());
   if (0 == file) {
      std::cerr <<"failed to open '"<<fileName<<"'";
      return 1;
   }
   
   TTree* eventTree = dynamic_cast<TTree*> (file->Get("Events"));
   
   if(0 == eventTree) {
      std::cerr <<"The file '"<<fileName<<"' does not contain an 'Events' TTree";
      return 1;
   }
   
   TBranch* branch = eventTree->GetBranch(branchName.c_str());
   
   if(0==branch) {
      std::cerr <<"The Events TTree does not contain the branch "<<branchName;
      return 1;
   }
   
   
   FWLiteEnabler::enable();

   TClass* cls = TClass::GetClass(branch->GetClassName());
   if(0==cls) {
      std::cerr <<"class '"<<branch->GetClassName()<<"' is unknown by ROOT\n";
      return 1;
   }
   
   void* objInstance = cls->New();
   if(0==objInstance) {
      std::cerr <<"unable to create a default instance of the class "<<branch->GetClassName();
      return 1;
   }

   //associate this with the branch
   void* pObjInstance = &objInstance;
   
   branch->SetAddress(pObjInstance);
   
   branch->GetEntry(entryNumber);
   
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
