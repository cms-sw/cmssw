#include "TFile.h"
#include "TKey.h"
#include "TTree.h"
#include "TError.h"
#include "TCollection.h"
#include <iostream>

void branchlist()
{
  std::string fname;
  std::string tname;
  gErrorIgnoreLevel = kError;
  while(true) {
    std::cin >> fname;
    if(fname == "quit")  return;
    std::cin >> tname;
    std::cout << "\nAll branches for TTree " << tname
              << " in file " << fname << "\n" << std::endl;
    TFile * file = new TFile(fname.c_str(),"READ","Test file");
    if(file) {
      TTree * tree;
      tname = tname + ";";
      file->GetObject(tname.c_str(),tree);
      if (tree)  {
	tree->Print("all");
      } else {
	std::cout << "There is no TTree object named " << tname << std::endl;
      }
    }
  }
}
