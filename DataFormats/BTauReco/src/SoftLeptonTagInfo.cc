#include <vector>

#include "FWCore/Utilities/interface/EDMException.h"

#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/Jet.h"

namespace reco {

using namespace btau;

const float SoftLeptonProperties::undefQuality = -999.0;

float SoftLeptonProperties::quality(unsigned int index, bool throwIfUndefined) const
{
  float qual = undefQuality;
  if (index < qualities_.size())
    qual = qualities_[index];

  if (qual == undefQuality && throwIfUndefined)
    throw edm::Exception(edm::errors::InvalidReference)
      << "Requested lepton quality not found in SoftLeptonProperties::quality"
      << std::endl;

  return qual;
}
  
void SoftLeptonProperties::setQuality(unsigned int index, float qual)
{
  if (qualities_.size() < index)
    qualities_.resize(index + 1, undefQuality);

  qualities_[index] = qual;
}
  
TaggingVariableList SoftLeptonTagInfo::taggingVariables(void) const {
  TaggingVariableList list;

  const Jet & jet = *( this->jet() );
  list.insert( TaggingVariable(jetEnergy, jet.energy()), true );
  list.insert( TaggingVariable(jetPt,     jet.et()),     true );
  list.insert( TaggingVariable(jetEta,    jet.eta()),    true );
  list.insert( TaggingVariable(jetPhi,    jet.phi()),    true );

  for (unsigned int i = 0; i < m_leptons.size(); ++i) {
    const Track & track = *(m_leptons[i].first);
    list.insert( TaggingVariable(trackMomentum,  track.p()),     true );
    list.insert( TaggingVariable(trackEta,       track.eta()),   true );
    list.insert( TaggingVariable(trackPhi,       track.phi()),   true );
    list.insert( TaggingVariable(trackChi2,      track.normalizedChi2()), true );
    const SoftLeptonProperties & data = m_leptons[i].second;
    list.insert( TaggingVariable(leptonQuality , data.quality()), true );
    list.insert( TaggingVariable(trackSip2dSig,  data.sip2d),    true );
    list.insert( TaggingVariable(trackSip3dSig,  data.sip3d),    true );
    list.insert( TaggingVariable(trackPtRel,     data.ptRel),    true );
    list.insert( TaggingVariable(trackP0Par,     data.ptRel),    true );
    list.insert( TaggingVariable(trackEtaRel,    data.etaRel),   true );
    list.insert( TaggingVariable(trackDeltaR,    data.deltaR),   true );
    list.insert( TaggingVariable(trackPParRatio, data.ratioRel), true );
  }

  list.finalize();
  return list;
}

}

