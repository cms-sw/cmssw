/*

This root macro takes a file name and a space separated list of other file names
and compares their modification times. If the first file is the most recently
modified, 0 is returned; otherwise, 1 is returned. If the first file is not
found, 1 is returned.

*/

#include "TROOT.h"
#include "TString.h"
#include "TObjString.h"

int compareFileAges(const char* newestCandidate, const char* filesToCompare) {

  Long_t dummy = 0;
  Long_t candidateTime = 0;
  Long_t comparisonTime = 0;
  int found = !gSystem->GetPathInfo(newestCandidate, &dummy, &dummy, &dummy,
                                   &candidateTime);
  //cout << newestCandidate << ": " << candidateTime << endl;
  // If the first file is not found, return 1
  if(!found)
    return 1;

  // Separate files in the list into an array
  TObjArray* compareList = TString(filesToCompare).Tokenize(" ");

  // Go through the array
  for (Int_t iFile = 0; iFile < compareList->GetEntriesFast(); ++iFile) {
    found = !gSystem->GetPathInfo(compareList->At(iFile)->GetName(), &dummy,
                                 &dummy, &dummy, &comparisonTime);
    //cout << compareList->At(iFile)->GetName() << ": " << comparisonTime << endl;
    // If the first file doesn't have the biggest modification time, return 1
    if(found && candidateTime <= comparisonTime)
      return 1;
  }

  // The first file had the biggest modification time: return 0
  return 0;
}
