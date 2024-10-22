#include <vector>
#include <cstring>

#include "FWCore/Utilities/interface/EDMException.h"

#include "DataFormats/BTauReco/interface/TemplatedSoftLeptonTagInfo.h"

namespace reco {

  using namespace btau;

  const float SoftLeptonProperties::Quality::undef = -999.0;

  unsigned int SoftLeptonProperties::Quality::internalByName(const char *name) {
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
        << "Requested lepton quality \"" << name << "\" not found in SoftLeptonProperties::Quality::byName"
        << std::endl;
  }

  float SoftLeptonProperties::quality(unsigned int index, bool throwIfUndefined) const {
    float qual = Quality::undef;
    if (index < qualities_.size())
      qual = qualities_[index];

    if (qual == Quality::undef && throwIfUndefined)
      throw edm::Exception(edm::errors::InvalidReference)
          << "Requested lepton quality not found in SoftLeptonProperties::Quality" << std::endl;

    return qual;
  }

  void SoftLeptonProperties::setQuality(unsigned int index, float qual) {
    if (qualities_.size() <= index)
      qualities_.resize(index + 1, Quality::undef);

    qualities_[index] = qual;
  }

}  // namespace reco
