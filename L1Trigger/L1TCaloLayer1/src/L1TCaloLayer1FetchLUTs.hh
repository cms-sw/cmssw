#ifndef L1TCaloLayer1FetchLUTs_hh
#define L1TCaloLayer1FetchLUTs_hh

#include "UCTGeometry.hh"
#include "FWCore/Framework/interface/EventSetup.h"
#include <vector>
#include <array>

// External function declaration



bool L1TCaloLayer1FetchLUTs(const edm::EventSetup& iSetup, 
			    std::vector< std::array< std::array< std::array<uint32_t, l1tcalo::nEtBins>, l1tcalo::nCalSideBins >, l1tcalo::nCalEtaBins> > &eLUT,
			    std::vector< std::array< std::array< std::array<uint32_t, l1tcalo::nEtBins>, l1tcalo::nCalSideBins >, l1tcalo::nCalEtaBins> > &hLUT,
                            std::vector< std::array< std::array<uint32_t, l1tcalo::nEtBins>, l1tcalo::nHfEtaBins > > &hfLUT,
                            std::vector<unsigned int> &ePhiMap,
                            std::vector<unsigned int> &hPhiMap,
                            std::vector<unsigned int> &hfPhiMap,
			    bool useLSB = true,
			    bool useCalib = true,
			    bool useECALLUT = true,
			    bool useHCALLUT = true,
                            bool useHFLUT = true,
                            int fwVersion = 0);

#endif
