#include <string>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"

#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"

using namespace btag;

LeptonSelector::LeptonSelector(const edm::ParameterSet &params) :
  m_sign(option(params.getParameter<std::string>("ipSign"))),
  m_leptonId(reco::SoftLeptonProperties::quality::btagLeptonCands),
  m_qualityCut(0.5)
{
  if (params.exists("leptonId") || params.exists("qualityCut")) {
    std::string leptonId = params.getParameter<std::string>("leptonId");
    m_leptonId = reco::SoftLeptonProperties::quality::byName<reco::SoftLeptonProperties::quality::Generic>(leptonId.c_str());
    m_qualityCut = params.getParameter<double>("qualityCut");
  }
}

LeptonSelector::~LeptonSelector()
{
}

bool LeptonSelector::operator() (const reco::SoftLeptonProperties &properties, bool use3d) const
{
  float sip = use3d ? properties.sip3d : properties.sip2d;
  if ((isPositive() && sip <= 0.0) ||
      (isNegative() && sip >= 0.0))
    return false;

  bool candSelection = (m_leptonId == reco::SoftLeptonProperties::quality::btagLeptonCands);
  float quality = properties.quality(m_leptonId, !candSelection);
  if (candSelection && quality == reco::SoftLeptonProperties::quality::undef)
   return true;		// for backwards compatibility

  return quality > m_qualityCut;
}

LeptonSelector::sign LeptonSelector::option(const std::string & selection)
{
  if (selection == "any")
    return any;
  else if (selection == "negative")
    return negative;
  else if (selection == "positive")
    return positive;
  else 
    throw edm::Exception( edm::errors::Configuration ) << "invalid parameter specified for soft lepton selection";
}
