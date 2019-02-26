#ifndef L1Trigger_Phase2L1ParticleFlow_PuppiAlgo_h
#define L1Trigger_Phase2L1ParticleFlow_PuppiAlgo_h

#include "L1Trigger/Phase2L1ParticleFlow/interface/PUAlgoBase.h"

namespace l1tpf_impl { 

  class PuppiAlgo : public PUAlgoBase {
    public:
        PuppiAlgo( const edm::ParameterSet& ) ;
        virtual ~PuppiAlgo() ;

        const std::vector<std::string> & puGlobalNames() const override ;
        void doPUGlobals(const std::vector<Region> &rs, float npu, std::vector<float> & globals) const override ;
        void runNeutralsPU(Region &r, float npu, const std::vector<float> & globals) const override ;

    protected:
        void computePuppiMedRMS(const std::vector<Region> &rs, float &alphaCMed, float &alphaCRms, float &alphaFMed, float &alphaFRms) const ;
        void fillPuppi(Region &r) const ;
        void computePuppiWeights(Region &r, float alphaCMed, float alphaCRms, float alphaFMed, float alphaFRms) const ;
        
        float puppiDr_, puppiDrMin_, puppiPtMax_;
        std::vector<float> puppiEtaCuts_, puppiPtCuts_, puppiPtCutsPhotons_;
        std::vector<int16_t> intPuppiEtaCuts_, intPuppiPtCuts_, intPuppiPtCutsPhotons_;
        bool puppiUsingBareTracks_;
        
  };

} // end namespace

#endif
