// Implementation of class LXXXCorrector.
// Generic LX jet corrector class.

#include "JetMETCorrections/Algorithms/interface/LXXXCorrector.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"


using namespace std;


//------------------------------------------------------------------------ 
//--- LXXXCorrector constructor ------------------------------------------
//------------------------------------------------------------------------
LXXXCorrector::LXXXCorrector(const edm::ParameterSet& fConfig) 
{
  string level(fConfig.getParameter<string>("level"));
  string era(fConfig.getParameter<string>("era"));
  string algorithm(fConfig.getParameter<string>("algorithm"));
  string section(fConfig.getUntrackedParameter<string>("section",""));  
  bool   debug (fConfig.getUntrackedParameter<bool>("debug",false));
  
  string fileName("CondFormats/JetMETObjects/data/");

  if (!era.empty()) fileName += era + "_";
  fileName += level;
  if (!algorithm.empty()) fileName += "_" + algorithm;
  fileName += ".txt";
  if (debug)
    edm::LogInfo("FileName")<<"initialize from "<<fileName;
    
  edm::FileInPath fip(fileName); 

  if (level == "L1Offset")
    {
      mLevel = 1;
      mCorrector = new FactorizedJetCorrector("L1",fip.fullPath());
    }
  else if (level == "L2Relative")
    {
      mLevel = 2;
      mCorrector = new FactorizedJetCorrector("L2",fip.fullPath());
    }
  else if (level == "L3Absolute")
    {
      mLevel = 3;  
      mCorrector = new FactorizedJetCorrector("L3",fip.fullPath());
    }
  else if (level == "L4EMF")
    {
      mLevel = 4;
      mCorrector = new FactorizedJetCorrector("L4",fip.fullPath());
    }
  else if (level == "L5Flavor")
    {
      mLevel = 5;
      string option = "Flavor:"+section;
      mCorrector = new FactorizedJetCorrector("L5",fip.fullPath(),option);
    }
  else if (level == "L7Parton")
    {
      mLevel = 7;
      string option = "Parton:"+section;
      mCorrector = new FactorizedJetCorrector("L7",fip.fullPath(),option);
    }
  else
    throw cms::Exception("LXXXCorrector")<<" unknown correction level "<<level; 
}
//------------------------------------------------------------------------ 
//--- LXXXCorrector destructor -------------------------------------------
//------------------------------------------------------------------------
LXXXCorrector::~LXXXCorrector() 
{
  delete mCorrector;
} 
//------------------------------------------------------------------------ 
//--- Returns correction for a given 4-vector ----------------------------
//------------------------------------------------------------------------
double LXXXCorrector::correction(const LorentzVector& fJet) const 
{
  // L1Offset correction is binned in eta and is a function of energy
  if (mLevel == 1)
    {
      mCorrector->setJetEta(fJet.eta()); 
      mCorrector->setJetE(fJet.energy());
    }
  // L2,L3,L5,L7 corrections are binned in eta and are a function of pt
  else if (mLevel == 2 || mLevel == 3 || mLevel == 5 || mLevel == 7)
    {
      mCorrector->setJetEta(fJet.eta()); 
      mCorrector->setJetPt(fJet.pt());
    }
  // L4 correction requires more information that a simple 4-vector
  else if (mLevel == 4)
    {
      throw cms::Exception("Invalid jet type") << "L4EMFCorrection is applicable to CaloJets only";
      return 1;
    }
  return mCorrector->getCorrection();
}
//------------------------------------------------------------------------ 
//--- Returns correction for a given jet ---------------------------------
//------------------------------------------------------------------------
double LXXXCorrector::correction(const reco::Jet& fJet) const 
{
  double result = 1.;
  // L4 correction applies to Calojets only
  if (mLevel == 4)
    {
      const reco::CaloJet& caloJet = dynamic_cast <const reco::CaloJet&> (fJet);
      mCorrector->setJetEta(fJet.eta()); 
      mCorrector->setJetPt(fJet.pt());
      mCorrector->setJetEMF(caloJet.emEnergyFraction());
      result = mCorrector->getCorrection();
    }
  else
    result = correction(fJet.p4());
  return result;
}
