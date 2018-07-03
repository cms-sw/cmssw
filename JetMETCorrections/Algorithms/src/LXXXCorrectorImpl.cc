// Implementation of class LXXXCorrectorImpl.
// Generic LX jet corrector class.

#include "JetMETCorrections/Algorithms/interface/LXXXCorrectorImpl.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrectorCalculator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/JPTJet.h"

using namespace std;


LXXXCorrectorImplMaker::LXXXCorrectorImplMaker(edm::ParameterSet const& fConfig, edm::ConsumesCollector):
JetCorrectorImplMakerBase(fConfig)
{
}

std::unique_ptr<reco::JetCorrectorImpl> 
LXXXCorrectorImplMaker::make(edm::Event const&, edm::EventSetup const& fSetup)
{
  unsigned int level =0;
  auto calculator = getCalculator(fSetup,
				  [&level](std::string const& levelName) 
    {
      if (levelName == "L2Relative")
	level = 2;
      else if (levelName == "L3Absolute")
	level = 3;  
      else if (levelName == "L4EMF")
	level = 4;
      else if (levelName == "L5Flavor")
	level = 5;
      else if (levelName == "L7Parton")
	level = 7;
      else if (levelName == "L2L3Residual")
        level = 8;
      else
	throw cms::Exception("LXXXCorrectorImpl")<<" unknown correction level "<<levelName;
    });
  return std::unique_ptr<reco::JetCorrectorImpl>(new LXXXCorrectorImpl(calculator,level));
}

void 
LXXXCorrectorImplMaker::fillDescriptions(edm::ConfigurationDescriptions& iDescriptions)
{
  edm::ParameterSetDescription desc;
  addToDescription(desc);

  iDescriptions.addDefault(desc);
}

//------------------------------------------------------------------------ 
//--- LXXXCorrectorImpl constructor ------------------------------------------
//------------------------------------------------------------------------
LXXXCorrectorImpl::LXXXCorrectorImpl(std::shared_ptr<FactorizedJetCorrectorCalculator const> calculator, unsigned int level):
  mLevel(level),
  mCorrector(calculator)
{
}
//------------------------------------------------------------------------ 
//--- Returns correction for a given 4-vector ----------------------------
//------------------------------------------------------------------------
double LXXXCorrectorImpl::correction(const LorentzVector& fJet) const 
{
  // L4 correction requires more information than a simple 4-vector
  if (mLevel == 4) {
    throw cms::Exception("Invalid jet type") << "L4EMFCorrection is applicable to CaloJets only";
    return 1;
  }

  FactorizedJetCorrectorCalculator::VariableValues values;
  values.setJetEta(fJet.eta()); 
  values.setJetE(fJet.energy());
  values.setJetPt(fJet.pt());
  values.setJetPhi(fJet.phi());

  return mCorrector->getCorrection(values);
}
//------------------------------------------------------------------------ 
//--- Returns correction for a given jet ---------------------------------
//------------------------------------------------------------------------
double LXXXCorrectorImpl::correction(const reco::Jet& fJet) const 
{
  double result = 1.;
  // L4 correction applies to Calojets only
  if (mLevel == 4) {
      const reco::CaloJet& caloJet = dynamic_cast <const reco::CaloJet&> (fJet);
      FactorizedJetCorrectorCalculator::VariableValues values;
      values.setJetEta(fJet.eta()); 
      values.setJetPt(fJet.pt());
      values.setJetEMF(caloJet.emEnergyFraction());
      result = mCorrector->getCorrection(values);
  }
  else
    result = correction(fJet.p4());
  return result;
}
