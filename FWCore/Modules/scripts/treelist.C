#include "TFile.h"
#include "TKey.h"
#include "TTree.h"
#include "TError.h"
#include "TCollection.h"
#include <iostream>

void treelist()
{
  std::string fname;
  gErrorIgnoreLevel = kError;
  std::cin >> fname;
  if(fname == "quit")  return;
  TObject * obj;
  TKey * key;
  TFile * file = new TFile(fname.c_str(),"READ","Test file");
  if(file) {
    std::cout << "\nNames of TTree objects in file: " << fname << std::endl;
    gErrorIgnoreLevel = kError;
    TIter next(file->GetListOfKeys());
    while( (key = (TKey*)next()) ) {
      obj = key->ReadObj();
      if ( (obj->InheritsFrom("TTree")) ) {
        std::cout << "\t\t" << ((TTree*)obj)->GetName() << std::endl;
      }
    }
  }
  std::cout << " " << std::endl;
}
