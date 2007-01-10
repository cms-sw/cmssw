#ifndef BTauReco_SoftLeptonTagInfo_h
#define BTauReco_SoftLeptonTagInfo_h

#include <iterator>

#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include "DataFormats/BTauReco/interface/JetTagProxy.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfoFwd.h"

namespace reco {
 
class SoftLeptonProperties {
public:
    enum {
        AXIS_ORIGINAL = 0,  // use the original (calorimietric) jet axis
        AXIS_CHARGED  = 1,  // refine jet axis from all charged tracks
        AXIS_EXCLUDED = 2   // refine, without the tagging lepton track
    };

    unsigned int axisRefinement;            // if and how the jet axis is refined
    double sip3d;                           // 3D signed inpact parameter
    double ptRel;                           // transverse momentum wrt. jet axis
    double etaRel;                          // (pseudo)rapidity along jet axis
    double deltaR;                          // pseudoangular distance to jet axis
    double ratio;                           // momentum over jet energy
    double ratioRel;                        // momentum parallet to jet axis over jet energy
    double tag;                             // discriminant using this track as tagging lepton
};

class SoftLeptonTagInfo : public JetTagProxy {
public:

    //typedef edm::AssociationMap< edm::OneToValue< TrackCollection, SoftLeptonProperties, unsigned int > > LeptonMap;
    typedef std::vector< std::pair< TrackRef, SoftLeptonProperties > > LeptonMap;
    
    SoftLeptonTagInfo(void) : m_leptons() {}

    virtual ~SoftLeptonTagInfo(void) {}
  
    virtual SoftLeptonTagInfo* clone(void)  const { return new SoftLeptonTagInfo(*this); }

    unsigned int leptons(void) const { 
        return m_leptons.size(); 
    } 

    TrackRef lepton(size_t i) const {
        // return find_iterator(i)->key;
        return m_leptons[i].first;
    }
    
    const SoftLeptonProperties& properties(size_t i) const {
        // return find_iterator(i)->val;
        return m_leptons[i].second;
    }

    void insert(TrackRef lepton, const SoftLeptonProperties& properties) {
        // m_leptons.insert( lepton, properties );
        m_leptons.push_back( std::pair< TrackRef, SoftLeptonProperties > (lepton, properties) );
    }

private:
    LeptonMap m_leptons;

    LeptonMap::const_iterator find_iterator(size_t i) const {
        LeptonMap::const_iterator it = m_leptons.begin();
        while (i--) {
            ++it;
            if (it == m_leptons.end())
                throw edm::Exception( edm::errors::InvalidReference, "lepton not found" );
        }
        return it;
    }
};

}

#endif // BTauReco_SoftLeptonTagInfo_h
