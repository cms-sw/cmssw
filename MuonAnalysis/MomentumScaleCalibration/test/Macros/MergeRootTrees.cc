#include <iostream>
#include <TFile.h>
#include <TChain.h>

/**
 * Simple skeleton of a macro acting as a reminder on how to merge root files.
 */

void MergeRootTrees()
{
  TChain * chain = new TChain("T");
  chain->Add("Background_1.root");
  chain->Add("Background_2.root");
  chain->Merge("Background.root");
}
