#ifndef L1Trigger_Phase2L1ParticleFlow_PFAlgoBase_h
#define L1Trigger_Phase2L1ParticleFlow_PFAlgoBase_h

#include <algorithm>

#include "L1Trigger/Phase2L1ParticleFlow/interface/Region.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace l1tpf_impl { 

  class PFAlgoBase {
    public:
        PFAlgoBase( const edm::ParameterSet& ) ;
        virtual ~PFAlgoBase() ;
        virtual void runPF(Region &r) const = 0;
    protected:
        int debug_;
        void initRegion(Region &r) const ;
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
