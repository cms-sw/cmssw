#include "TFile.h"
#include "TTree.h"
#include "TError.h"
#include <iostream>

void inventory()
{
  std::string filename;
  gErrorIgnoreLevel = kError;
  while(true) {
    std::cin >> filename;
    if(filename == "quit")  return;
    TFile * file = new TFile(filename.c_str(),"READ","Test file");
    if(file) {
      TTree * tree;
      file->GetObject("Events;",tree);
      if (tree)  tree->Print("all");
    }
  }
  return;
}
