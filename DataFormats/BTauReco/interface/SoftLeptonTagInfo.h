#ifndef BTauReco_BJetTagSoftLepton_h
#define BTauReco_BJetTagSoftLepton_h

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfoFwd.h"

namespace reco {
 
class SoftLeptonTagInfo {
public:
    SoftLeptonTagInfo(const JetTracksAssociation& jetTracks, 
                      const TrackRefVector&       leptons, 
                      const std::vector<double>&  leptonProbabilities, 
                      const std::vector<double>&  leptonTags);

    explicit SoftLeptonTagInfo(const JetTracksAssociation& jetTracks);
  
    virtual ~SoftLeptonTagInfo(void) {}
  
    virtual SoftLeptonTagInfo* clone(void)  const { return new SoftLeptonTagInfo(*this); }

    virtual float discriminator(void)       const { return m_discriminator; }
    virtual float discriminator(size_t n)   const { return (n < m_leptons.size()) ? m_jetTagProbability[n] : 0.; }
  
  
private:
    edm::Ref<JetTagCollection> m_jetTag;
    edm::RefVector<TrackCollection> m_leptons;
    std::vector<double> m_leptonProbability;     // probability of each track to be a lepton
    std::vector<double> m_leptonJetTag;          // discriminator using each track as lepton
 };

}

#endif // BTauReco_BJetTagSoftLepton_h
