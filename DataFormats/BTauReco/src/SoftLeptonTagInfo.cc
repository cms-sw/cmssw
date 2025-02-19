#include <vector>
#include <cstring>

#include "FWCore/Utilities/interface/EDMException.h"

#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/JetReco/interface/Jet.h"

namespace reco {

using namespace btau;

const float SoftLeptonProperties::quality::undef = -999.0;

unsigned int SoftLeptonProperties::quality::internalByName(const char *name)
{
  if (std::strcmp(name, "") == 0)
    return 0;

  if (std::strcmp(name, "leptonId") == 0)
    return leptonId;
  else if (std::strcmp(name, "btagLeptonCands") == 0)
    return btagLeptonCands;

  if (std::strcmp(name, "pfElectronId") == 0)
    return pfElectronId;
  else if (std::strcmp(name, "btagElectronCands") == 0)
    return btagElectronCands;

  if (std::strcmp(name, "muonId") == 0)
    return muonId;
  else if (std::strcmp(name, "btagMuonCands") == 0)
    return btagMuonCands;

  throw edm::Exception(edm::errors::Configuration) 
    << "Requested lepton quality \"" << name
    << "\" not found in SoftLeptonProperties::quality:byName"
    << std::endl;
}

float SoftLeptonProperties::quality(unsigned int index, bool throwIfUndefined) const
{
  float qual = quality::undef;
  if (index < qualities_.size())
    qual = qualities_[index];

  if (qual == quality::undef && throwIfUndefined)
    throw edm::Exception(edm::errors::InvalidReference)
      << "Requested lepton quality not found in SoftLeptonProperties::quality"
      << std::endl;

  return qual;
}
  
void SoftLeptonProperties::setQuality(unsigned int index, float qual)
{
  if (qualities_.size() <= index)
    qualities_.resize(index + 1, quality::undef);

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
    list.insert( TaggingVariable(leptonQuality,  data.quality(SoftLeptonProperties::quality::leptonId, false)), true );
    list.insert( TaggingVariable(leptonQuality2, data.quality(SoftLeptonProperties::quality::btagLeptonCands, false)), true );
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

