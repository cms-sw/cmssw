// Implementation of class LXXXCorrector.
// Generic LX jet corrector class.

#include "JetMETCorrections/Algorithms/interface/LXXXCorrector.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/JetReco/interface/JPTJet.h"

using namespace std;


//------------------------------------------------------------------------ 
//--- LXXXCorrector constructor ------------------------------------------
//------------------------------------------------------------------------
LXXXCorrector::LXXXCorrector(const JetCorrectorParameters& fParam, const edm::ParameterSet& fConfig) 
{
  string level = fParam.definitions().level();
  if (level == "L1Offset")
    mLevel = 1;
  else if (level == "L2Relative")
    mLevel = 2;
  else if (level == "L3Absolute")
    mLevel = 3;  
  else if (level == "L4EMF")
    mLevel = 4;
  else if (level == "L5Flavor")
    mLevel = 5;
  else if (level == "L7Parton")
    mLevel = 7;
  else if (level == "L1JPTOffset")
    mLevel = 8;
  else
    throw cms::Exception("LXXXCorrector")<<" unknown correction level "<<level; 
  vector<JetCorrectorParameters> vParam;
  vParam.push_back(fParam);
  mCorrector = new FactorizedJetCorrector(vParam);
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
  // L4 correction requires more information that a simple 4-vector
  if (mLevel == 4)
    {
      throw cms::Exception("Invalid jet type") << "L4EMFCorrection is applicable to CaloJets only";
      return 1;
    }
  else
    {
      mCorrector->setJetEta(fJet.eta()); 
      mCorrector->setJetE(fJet.energy());
      mCorrector->setJetPt(fJet.pt());
      mCorrector->setJetPhi(fJet.phi());
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
  else if ( mLevel == 8 ) 
    {
      // For JPT offset only   
      const reco::JPTJet& jptjet = dynamic_cast <const reco::JPTJet&> (fJet);
      edm::RefToBase<reco::Jet> jptjetRef = jptjet.getCaloJetRef();
      double zspcor_old = jptjet.getZSPCor();
      reco::CaloJet const * rawcalojet = dynamic_cast<reco::CaloJet const *>( &* jptjetRef);

      mCorrector->setJetEta(rawcalojet->eta()); 
      mCorrector->setJetPt(rawcalojet->et());
      
      double aa = mCorrector->getCorrection();
      double resulto(0.0);
      if (aa != 1.0)
        resulto = 1./(1.-aa);
      double enew = jptjet.energy() - (zspcor_old-resulto)*rawcalojet->energy();
      result = enew/jptjet.energy();
    }
  else
    result = correction(fJet.p4());
  return result;
}
