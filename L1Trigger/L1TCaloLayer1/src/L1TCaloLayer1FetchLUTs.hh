#ifndef L1TCaloLayer1FetchLUTs_hh
#define L1TCaloLayer1FetchLUTs_hh
// External function declaration

bool L1TCaloLayer1FetchLUTs(const edm::EventSetup& iSetup, 
			    std::vector< std::vector< std::vector < uint32_t > > > &eLUT,
			    std::vector< std::vector< std::vector < uint32_t > > > &hLUT,
                            std::vector< std::vector< uint32_t > > &hfLUT,
			    bool useLSB = true,
			    bool useCalib = true,
			    bool useECALLUT = true,
			    bool useHCALLUT = true,
                            bool useHFLUT = true);

#endif
