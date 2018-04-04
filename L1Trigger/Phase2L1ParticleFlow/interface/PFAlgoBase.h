#ifndef L1Trigger_Phase2L1ParticleFlow_PFAlgoBase_h
#define L1Trigger_Phase2L1ParticleFlow_PFAlgoBase_h

#include <algorithm>

#include "L1Trigger/Phase2L1ParticleFlow/interface/Region.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace l1tpf_impl { 

  class PFAlgoBase {
    public:
        PFAlgoBase( const edm::ParameterSet& ) ;
        virtual void runPF(Region &r) const = 0;
        virtual void runChargedPV(Region &r, float z0) const ;
        virtual void runPuppi(Region &r, float npu, float alphaCMed, float alphaCRms, float alphaFMed, float alphaFRms) const ;
        /// global operations
        enum VertexAlgo { OldVtxAlgo, TPVtxAlgo };
        virtual void doVertexing(std::vector<Region> &rs, VertexAlgo algo, float &vz) const ; // region is not const since it sets the fromPV bit of the tracks
        virtual void computePuppiMedRMS(const std::vector<Region> &rs, float &alphaCMed, float &alphaCRms, float &alphaFMed, float &alphaFRms) const ;
    protected:
        float etaCharged_, puppiDr_; 
        std::vector<float> puppiEtaCuts_, puppiPtCuts_, puppiPtCutsPhotons_;
        std::vector<int16_t> intPuppiEtaCuts_, intPuppiPtCuts_, intPuppiPtCutsPhotons_;
        float vtxRes_;
        bool vtxAdaptiveCut_; 
        int debug_;
        void initRegion(Region &r) const ;
        void computePuppiWeights(Region &r, float alphaCMed, float alphaCRms, float alphaFMed, float alphaFRms) const ;
        void fillPuppi(Region &r) const ;
        PFParticle & addTrackToPF(Region &r, const PropagatedTrack &tk) const { return addTrackToPF(r.pf, tk); }
        PFParticle & addCaloToPF(Region &r, const CaloCluster &calo) const { return addCaloToPF(r.pf, calo); }
        //PFParticle & discardTrack(Region &r, const PropagatedTrack &tk, int status) const { 
        //    PFParticle & ret = addTrackToPF(r.pfdiscarded, tk); 
        //    ret.hwStatus = status;
        //    return ret;
        //}
        //PFParticle & discardCalo(Region &r, const CaloCluster &calo, int status) const { 
        //    PFParticle & ret = addCaloToPF(r.pfdiscarded, calo); 
        //    ret.hwStatus = status;
        //    return ret;
        //}
        PFParticle & addTrackToPF(std::vector<PFParticle> &pfs, const PropagatedTrack &tk) const ;
        PFParticle & addCaloToPF(std::vector<PFParticle> &pfs, const CaloCluster &calo) const ;
  };

} // end namespace

#endif
