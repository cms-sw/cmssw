#include "TFile.h"
#include "TKey.h"
#include "TTree.h"
#include "TError.h"
#include "TCollection.h"
#include <iostream>

void allbranches()
{
  char * charname;
  std::string fname;
  gErrorIgnoreLevel = kError;
  std::cin >> fname;
  if(fname == "quit")  return;
  TObject * obj;
  TTree * tree;
  TKey * key;
  TFile * file = new TFile(fname.c_str(),"READ","Test file");
  if(file) {
    TIter next(file->GetListOfKeys());
    while( (key = (TKey*)next()) ) {
      obj = key->ReadObj();
      if ( (obj->InheritsFrom("TTree")) ) {
        charname = ((TTree*)obj)->GetName();
        std::cout << "\nAll branches for TTree " << charname 
                  << " in file " << fname << "\n" << std::endl;
	file->GetObject(charname,tree);
        if(tree) {
          tree->Print("all");
        } else {
          std::cout << "Anomoly.  There is no tree named " << charname << std::endl;
        }
      }
    }
  }
  std::cout << " " << std::endl;
}
