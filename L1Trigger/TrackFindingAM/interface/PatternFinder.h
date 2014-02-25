#ifndef _PATTERNFINDER_H_
#define _PATTERNFINDER_H_

#include <iostream>
#include <fstream>
#include <TChain.h>
#include <TFile.h>
#include "SectorTree.h"

using namespace std;

/**
   \brief Used to find active patterns in events. Stubs of the event are injected in a virtual detector, then we count the number of active patterns in the bank (the patterns have previously been linked to the virtual detector's supertrips).
**/
class PatternFinder{
 private:
  int superStripSize;
  int active_threshold;
  SectorTree* sectors;
  string eventsFilename;
  string outputFileName;
  Detector tracker;

 public:
 /**
     \brief Constructor
     \param sp Size of a super strip
     \param at The minimum number of hit super strip to activate a pattern
     \param st The SectorTree containing the sectors with their associated patterns
     \param f The name of the file to analyse
     \param of The name of the output file
  **/
  PatternFinder(int sp, int at, SectorTree* st, string f, string of);
  /**
     \brief Set the SectorTree (contains sectors with their patterns)
     \param s The SectorTree containing the sectors with their associated patterns
  **/
  void setSectorTree(SectorTree* s);

  /**
     \brief Set the name of the root file containing events
     \param f The name of the file
  **/
  void setEventsFile(string f);
  /**
     \brief Look for active patterns in events
     \param start The search will start from this event number
     \param stop The search will end at this event number
  **/
  void find(int start, int& stop);

  /**
     \brief Get active patterns from list of hits (public for CMSSW).
  **/
  vector<Sector*> find(vector<Hit*> hits);

  /**
     \brief Merge 2 files into 1 single file
  **/
  static void mergeFiles(string outputFile, string inputFile1, string inputFile2);
};
#endif
