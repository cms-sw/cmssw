#ifndef OMTF_OMTFSorter_H
#define OMTF_OMTFSorter_H

#include <tuple>
#include <vector>

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include "L1Trigger/L1TMuonOverlap/interface/OMTFResult.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFProcessor.h"
#include "L1Trigger/L1TMuonOverlap/interface/AlgoMuon.h"
#include "L1Trigger/L1TMuonOverlap/interface/GhostBuster.h"

class OMTFSorter{

 public:

  void setNphiBins(unsigned int phiBins) { nPhiBins = phiBins;}

  void sortRefHitResults(const std::vector<OMTFProcessor::resultsMap> & procResults,
        std::vector<AlgoMuon> & refHitCleanCands,
        int charge=0);


  void sortProcessorAndFillCandidates(unsigned int iProcessor, l1t::tftype mtfType,
                 const std::vector<AlgoMuon> & algoCands,
                 l1t::RegionalMuonCandBxCollection & sortedCands,
                 int bx, int charge=0);


  ///Sort results from a single reference hit.
  ///Select candidate with highest number of hit layers
  ///Then select a candidate with largest likelihood value and given charge
  ///as we allow two candidates with opposite charge from single 10deg region
  AlgoMuon sortRefHitResults(const OMTFProcessor::resultsMap & aResultsMap,
				int charge=0);

 private:

  ///Check if the hit pattern of given OMTF candite is not on the list
  ///of invalid hit patterns. Invalid hit patterns provode very little
  ///to efficiency, but gives high contribution to rate.
  ///Candidate with invalid hit patterns is assigned quality=0.
  ///Currently the list of invalid patterns is hardcoded.
  ///This has to be read from configuration.
  bool checkHitPatternValidity(unsigned int hits);

  ///Find a candidate with best parameters for given GoldenPattern
  ///Sorting is made amongs candidates with different reference layers
  ///The output tuple contains (nHitsMax, pdfValMax, refPhi, refLayer, hitsWord, refEta)
  ///hitsWord codes number of layers hit: hitsWord= sum 2**iLogicLayer,
  ///where sum runs over layers which were hit
  AlgoMuon sortSingleResult(const OMTFResult & aResult);


  unsigned int nPhiBins;
};

#endif
