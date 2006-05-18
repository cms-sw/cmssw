#ifndef BTauReco_BJetTagSoftLepton_h
#define BTauReco_BJetTagSoftLepton_h

#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfoFwd.h"

namespace reco {
 
class SoftLeptonTagInfo
 {
  public:

  SoftLeptonTagInfo() {}
  virtual ~SoftLeptonTagInfo() {}
  
  virtual float discriminator() const           { return m_discriminator; }
  virtual float discriminator(size_t n) const   { return (n < m_leptons.size()) ? m_jetTagProbability[n] : 0; }


  
  virtual SoftLeptonTagInfo* clone() const { return new SoftLeptonTagInfo(*this); }
  
  private:
   edm::Ref<JetTagCollection> m_jetTag;
   edm::RefVector<TrackCollection> m_leptons;
   std::vector<double> m_leptonProbability;     // probability of each track to be a lepton
   std::vector<double> m_jetTagProbability;     // discriminator using each track as lepton
 };

}
#endif
