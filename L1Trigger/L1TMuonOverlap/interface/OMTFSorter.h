#ifndef OMTF_OMTFSorter_H
#define OMTF_OMTFSorter_H

#include <tuple>
#include <vector>

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCandFwd.h"

#include "L1Trigger/L1TMuonOverlap/interface/OMTFResult.h"
#include "L1Trigger/L1TMuonOverlap/interface/OMTFProcessor.h"
#include "L1Trigger/L1TMuonOverlap/interface/InternalObj.h"


class OMTFSorter{

 public:

  ///Sort all processor results.
  ///First for each region cone find a best candidate using sortRegionResults()
  ///Then select best candidate amongs found for each logic region.
  ///The sorting is made for candidates with a given charge
  InternalObj sortProcessorResults(const std::vector<OMTFProcessor::resultsMap> & procResults,
				   int charge=0);
  //
  void sortProcessorResults(const std::vector<OMTFProcessor::resultsMap> & procResults,
			    std::vector<InternalObj> & refHitCleanCands,
			    int charge=0);

  ///Sort all processor results.
  ///First for each region cone find a best candidate using sortRegionResults()
  ///Then select best candidate amongs found for each logic region
  l1t::RegionalMuonCand sortProcessor(const std::vector<OMTFProcessor::resultsMap> & procResults,
						int charge=0);
  //
  void sortProcessor(const std::vector<OMTFProcessor::resultsMap> & procResults,
		     l1t::RegionalMuonCandBxCollection & sortedCands,
		     int bx, int charge=0);

  ///Sort results from a single reference hit.
  ///Select candidate with highest number of hit layers
  ///Then select a candidate with largest likelihood value and given charge
  ///as we allow two candidates with opposite charge from single 10deg region
  InternalObj sortRefHitResults(const OMTFProcessor::resultsMap & aResultsMap,
				int charge=0);

 private:

  ///Find a candidate with best parameters for given GoldenPattern
  ///Sorting is made amongs candidates with different reference layers
  ///The output tuple contains (nHitsMax, pdfValMax, refPhi, refLayer, hitsWord, refEta)
  ///hitsWord codes number of layers hit: hitsWord= sum 2**iLogicLayer,
  ///where sum runs over layers which were hit
  std::tuple<unsigned int,unsigned int, int, int, unsigned int, int> sortSingleResult(const OMTFResult & aResult);

};

#endif
