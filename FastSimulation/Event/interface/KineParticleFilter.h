#ifndef FastSimulation_Event_KineParticleFilter_H
#define FastSimulation_Event_KineParticleFilter_H

#include <set>
#include "DataFormats/Math/interface/LorentzVector.h"

class RawParticle;
namespace edm { 
  class ParameterSet;
}

class KineParticleFilter
{
public:
    KineParticleFilter(const edm::ParameterSet& kine); 
    virtual ~KineParticleFilter(){;};
    
    void setMainVertex(const math::XYZTLorentzVector& mv) { mainVertex=mv; }
    
    const math::XYZTLorentzVector& vertex() const { return mainVertex; }
    
    virtual bool accept(const RawParticle & p) const;

private:
    
    double etaMin, etaMax, phiMin, phiMax, pTMin, pTMax, EMin, EMax;
    double cos2Max, cos2PreshMin, cos2PreshMax;
    math::XYZTLorentzVector mainVertex;
    
    std::set<int>   forbiddenPdgCodes;
};

#endif
