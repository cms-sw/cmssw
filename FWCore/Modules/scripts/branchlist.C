#include "TFile.h"
#include "TKey.h"
#include "TTree.h"
#include "TError.h"
#include "TCollection.h"
#include <iostream>

void branchlist()
{
  std::string filename;
  std::string treename;
  gErrorIgnoreLevel = kError;
  while(true) {
    std::cin >> filename;
    std::cin >> treename;
    if(filename == "quit")  return;
    TFile * file = new TFile(filename.c_str(),"READ","Test file");
    if(file) {
      TTree * tree;
      treename = treename + ";";
      file->GetObject(treename.c_str(),tree);
      if (tree)  {
	tree->Print("all");
      } else {
	std::cout << "There is no TTree object named " << treename << std::endl;
      }
    }
  }
}
