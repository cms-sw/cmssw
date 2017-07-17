#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iomanip>

#include "TFile.h"
#include "TTree.h"
#include "TList.h"
#include "TIterator.h"

void print_rates(TFile * file) {
  TTree * tree = (TTree *) file->Get("l1uGTTree/L1uGTTree");

  // extract the list of Aliases
  std::vector<std::string> names;
  TList * aliases = tree->GetListOfAliases();
  TIter iter(aliases);
  std::for_each(iter.Begin(), TIter::End(), [&](TObject* alias){ names.push_back(alias->GetName()); } );

  unsigned long long entries = tree->GetEntries();
  if (entries == 0)
    return;

  int digits = std::log(entries) / std::log(10) + 1;
  for (auto const & name: names) {
    unsigned long long counts = tree->GetEntries(name.c_str());
    std::cout << std::setw(digits) << counts << " / " << std::setw(digits) << entries << "  " << name << std::endl;
  }
}


int main(int argc, char** argv) {
  if (argc < 2)
    return 0;

  TFile * file = TFile::Open(argv[1]);
  print_rates(file);
}
