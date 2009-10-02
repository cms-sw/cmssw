#ifndef TrackerAlignment_TreeMerger_H
#define TrackerAlignment_TreeMerger_H


#include <Riostream.h>
#include <string>
#include <fstream>
#include <map>

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventPrincipal.h" 
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"


#include "TFile.h"
#include "TString.h"
#include "TChain.h"
#include "TStopwatch.h"

class TreeMerger : public edm::EDAnalyzer{

 public:
  TreeMerger(const edm::ParameterSet &iConfig);
  ~TreeMerger();
  void beginJob( const edm::EventSetup &iSetup);
  void endJob();
  void analyze(const edm::Event&, const edm::EventSetup&);

 private:
  TTree *out_;//TTree containing the merged result
  TTree *firsttree_;//first tree of the list; this gives the structure to all the others 
  TChain *ch_;//chain containing all the tree you want to merge
  std::string filelist_;//text file containing the list of input files
  std::string firstfilename_;
  std::string treename_;//name of the tree you want to merge (contained in the file)
  std::string outfilename_;//name of the file where you want to save the output
 
  //Hit Population
  typedef map<uint32_t,uint32_t>DetHitMap;
  DetHitMap hitmap_;
  DetHitMap overlapmap_;
  int maxhits_;//above this number, the hit population is prescaled. Configurable for each subdet 
  edm::ParameterSet maxhitsSet_;
  int maxPXBhits_, maxPXFhits_, maxTIBhits_, maxTIDhits_, maxTOBhits_, maxTEChits_;
 

  TStopwatch myclock;

};




#endif
