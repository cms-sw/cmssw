#include "DQMOffline/Lumi/interface/ElectronIdentifier.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Utilities/interface/RegexMatch.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"

#include "DQMOffline/Lumi/interface/TTrigger.h"
#include "DQMOffline/Lumi/interface/TriggerTools.h"

#include <TLorentzVector.h>
#include <TMath.h>
#include <algorithm>

ElectronIdentifier::ElectronIdentifier (const edm::ParameterSet& c) :
   _effectiveAreas( (c.getParameter<edm::FileInPath>("effAreasConfigFile")).fullPath())

{
   rho_ = -1;
   ID_ = -1;
   cuts_[EleIDCutNames::SIGMAIETA][EleIDWorkingPoints::VETO]  [EleIDEtaBins::BARREL] = 0.0115;
   cuts_[EleIDCutNames::SIGMAIETA][EleIDWorkingPoints::LOOSE] [EleIDEtaBins::BARREL] = 0.011;
   cuts_[EleIDCutNames::SIGMAIETA][EleIDWorkingPoints::MEDIUM][EleIDEtaBins::BARREL] = 0.00998;
   cuts_[EleIDCutNames::SIGMAIETA][EleIDWorkingPoints::TIGHT] [EleIDEtaBins::BARREL] = 0.00998;

   cuts_[EleIDCutNames::SIGMAIETA][EleIDWorkingPoints::VETO]  [EleIDEtaBins::ENDCAP] = 0.037;
   cuts_[EleIDCutNames::SIGMAIETA][EleIDWorkingPoints::LOOSE] [EleIDEtaBins::ENDCAP] = 0.0314;
   cuts_[EleIDCutNames::SIGMAIETA][EleIDWorkingPoints::MEDIUM][EleIDEtaBins::ENDCAP] = 0.0298;
   cuts_[EleIDCutNames::SIGMAIETA][EleIDWorkingPoints::TIGHT] [EleIDEtaBins::ENDCAP] = 0.0292;

   cuts_[EleIDCutNames::DETAINSEED][EleIDWorkingPoints::VETO]  [EleIDEtaBins::BARREL] = 0.00749;
   cuts_[EleIDCutNames::DETAINSEED][EleIDWorkingPoints::LOOSE] [EleIDEtaBins::BARREL] = 0.00477;
   cuts_[EleIDCutNames::DETAINSEED][EleIDWorkingPoints::MEDIUM][EleIDEtaBins::BARREL] = 0.00311;
   cuts_[EleIDCutNames::DETAINSEED][EleIDWorkingPoints::TIGHT] [EleIDEtaBins::BARREL] = 0.00308;

   cuts_[EleIDCutNames::DETAINSEED][EleIDWorkingPoints::VETO]  [EleIDEtaBins::ENDCAP] = 0.00895;
   cuts_[EleIDCutNames::DETAINSEED][EleIDWorkingPoints::LOOSE] [EleIDEtaBins::ENDCAP] = 0.00868;
   cuts_[EleIDCutNames::DETAINSEED][EleIDWorkingPoints::MEDIUM][EleIDEtaBins::ENDCAP] = 0.00609;
   cuts_[EleIDCutNames::DETAINSEED][EleIDWorkingPoints::TIGHT] [EleIDEtaBins::ENDCAP] = 0.00605;

   cuts_[EleIDCutNames::DPHIIN][EleIDWorkingPoints::VETO]  [EleIDEtaBins::BARREL] = 0.228;
   cuts_[EleIDCutNames::DPHIIN][EleIDWorkingPoints::LOOSE] [EleIDEtaBins::BARREL] = 0.222;
   cuts_[EleIDCutNames::DPHIIN][EleIDWorkingPoints::MEDIUM][EleIDEtaBins::BARREL] = 0.103;
   cuts_[EleIDCutNames::DPHIIN][EleIDWorkingPoints::TIGHT] [EleIDEtaBins::BARREL] = 0.0816;

   cuts_[EleIDCutNames::DPHIIN][EleIDWorkingPoints::VETO]  [EleIDEtaBins::ENDCAP] = 0.213;
   cuts_[EleIDCutNames::DPHIIN][EleIDWorkingPoints::LOOSE] [EleIDEtaBins::ENDCAP] = 0.213;
   cuts_[EleIDCutNames::DPHIIN][EleIDWorkingPoints::MEDIUM][EleIDEtaBins::ENDCAP] = 0.045;
   cuts_[EleIDCutNames::DPHIIN][EleIDWorkingPoints::TIGHT] [EleIDEtaBins::ENDCAP] = 0.0394;

   cuts_[EleIDCutNames::HOVERE][EleIDWorkingPoints::VETO]  [EleIDEtaBins::BARREL] = 0.356;
   cuts_[EleIDCutNames::HOVERE][EleIDWorkingPoints::LOOSE] [EleIDEtaBins::BARREL] = 	0.298;
   cuts_[EleIDCutNames::HOVERE][EleIDWorkingPoints::MEDIUM][EleIDEtaBins::BARREL] = 0.253;
   cuts_[EleIDCutNames::HOVERE][EleIDWorkingPoints::TIGHT] [EleIDEtaBins::BARREL] = 0.0414;

   cuts_[EleIDCutNames::HOVERE][EleIDWorkingPoints::VETO]  [EleIDEtaBins::ENDCAP] = 0.211;
   cuts_[EleIDCutNames::HOVERE][EleIDWorkingPoints::LOOSE] [EleIDEtaBins::ENDCAP] = 0.101;
   cuts_[EleIDCutNames::HOVERE][EleIDWorkingPoints::MEDIUM][EleIDEtaBins::ENDCAP] = 0.0878;
   cuts_[EleIDCutNames::HOVERE][EleIDWorkingPoints::TIGHT] [EleIDEtaBins::ENDCAP] = 0.0641;

   cuts_[EleIDCutNames::ISO][EleIDWorkingPoints::VETO]  [EleIDEtaBins::BARREL] = 0.175;
   cuts_[EleIDCutNames::ISO][EleIDWorkingPoints::LOOSE] [EleIDEtaBins::BARREL] = 0.0994;
   cuts_[EleIDCutNames::ISO][EleIDWorkingPoints::MEDIUM][EleIDEtaBins::BARREL] = 0.0695;
   cuts_[EleIDCutNames::ISO][EleIDWorkingPoints::TIGHT] [EleIDEtaBins::BARREL] = 0.0588;

   cuts_[EleIDCutNames::ISO][EleIDWorkingPoints::VETO]  [EleIDEtaBins::ENDCAP] = 0.159;
   cuts_[EleIDCutNames::ISO][EleIDWorkingPoints::LOOSE] [EleIDEtaBins::ENDCAP] = 0.107;
   cuts_[EleIDCutNames::ISO][EleIDWorkingPoints::MEDIUM][EleIDEtaBins::ENDCAP] = 0.0821;
   cuts_[EleIDCutNames::ISO][EleIDWorkingPoints::TIGHT] [EleIDEtaBins::ENDCAP] = 0.0571;

   cuts_[EleIDCutNames::ONEOVERE][EleIDWorkingPoints::VETO]  [EleIDEtaBins::BARREL] = 0.299;
   cuts_[EleIDCutNames::ONEOVERE][EleIDWorkingPoints::LOOSE] [EleIDEtaBins::BARREL] = 0.241;
   cuts_[EleIDCutNames::ONEOVERE][EleIDWorkingPoints::MEDIUM][EleIDEtaBins::BARREL] = 0.134;
   cuts_[EleIDCutNames::ONEOVERE][EleIDWorkingPoints::TIGHT] [EleIDEtaBins::BARREL] = 0.0129;

   cuts_[EleIDCutNames::ONEOVERE][EleIDWorkingPoints::VETO]  [EleIDEtaBins::ENDCAP] = 0.15;
   cuts_[EleIDCutNames::ONEOVERE][EleIDWorkingPoints::LOOSE] [EleIDEtaBins::ENDCAP] = 0.14;
   cuts_[EleIDCutNames::ONEOVERE][EleIDWorkingPoints::MEDIUM][EleIDEtaBins::ENDCAP] = 0.13;
   cuts_[EleIDCutNames::ONEOVERE][EleIDWorkingPoints::TIGHT] [EleIDEtaBins::ENDCAP] = 0.0129;

   cuts_[EleIDCutNames::MISSINGHITS][EleIDWorkingPoints::VETO]  [EleIDEtaBins::BARREL] = 2;
   cuts_[EleIDCutNames::MISSINGHITS][EleIDWorkingPoints::LOOSE] [EleIDEtaBins::BARREL] = 1;
   cuts_[EleIDCutNames::MISSINGHITS][EleIDWorkingPoints::MEDIUM][EleIDEtaBins::BARREL] = 1;
   cuts_[EleIDCutNames::MISSINGHITS][EleIDWorkingPoints::TIGHT] [EleIDEtaBins::BARREL] = 1;

   cuts_[EleIDCutNames::MISSINGHITS][EleIDWorkingPoints::VETO]  [EleIDEtaBins::ENDCAP] = 3;
   cuts_[EleIDCutNames::MISSINGHITS][EleIDWorkingPoints::LOOSE] [EleIDEtaBins::ENDCAP] = 1;
   cuts_[EleIDCutNames::MISSINGHITS][EleIDWorkingPoints::MEDIUM][EleIDEtaBins::ENDCAP] = 1;
   cuts_[EleIDCutNames::MISSINGHITS][EleIDWorkingPoints::TIGHT] [EleIDEtaBins::ENDCAP] = 1;

   cuts_[EleIDCutNames::CONVERSION][EleIDWorkingPoints::VETO]  [EleIDEtaBins::BARREL] = 1;
   cuts_[EleIDCutNames::CONVERSION][EleIDWorkingPoints::LOOSE] [EleIDEtaBins::BARREL] = 1;
   cuts_[EleIDCutNames::CONVERSION][EleIDWorkingPoints::MEDIUM][EleIDEtaBins::BARREL] = 1;
   cuts_[EleIDCutNames::CONVERSION][EleIDWorkingPoints::TIGHT] [EleIDEtaBins::BARREL] = 1;

   cuts_[EleIDCutNames::CONVERSION][EleIDWorkingPoints::VETO]  [EleIDEtaBins::ENDCAP] = 1;
   cuts_[EleIDCutNames::CONVERSION][EleIDWorkingPoints::LOOSE] [EleIDEtaBins::ENDCAP] = 1;
   cuts_[EleIDCutNames::CONVERSION][EleIDWorkingPoints::MEDIUM][EleIDEtaBins::ENDCAP] = 1;
   cuts_[EleIDCutNames::CONVERSION][EleIDWorkingPoints::TIGHT] [EleIDEtaBins::ENDCAP] = 1;
}

void ElectronIdentifier::setRho(double rho) {
   if(rho >= 0) {
      rho_ = rho;
   } else {
      throw cms::Exception("ValueError")
         << "Encountered invalid value for energy density rho.\n"
         << "Value: " << rho << "\n"
         << "Rho should be a real, positive number.\n";
   }
}
void ElectronIdentifier::setID(std::string ID) {
   if(ID=="TIGHT") ID_ = EleIDWorkingPoints::TIGHT;
   else if(ID=="MEDIUM") ID_ = EleIDWorkingPoints::MEDIUM;
   else if(ID=="LOOSE") ID_ = EleIDWorkingPoints::LOOSE;
   else if(ID=="VETO") ID_ = EleIDWorkingPoints::VETO;
   else throw;
}
float ElectronIdentifier::dEtaInSeed(const reco::GsfElectronPtr& ele){
         return ele->superCluster().isNonnull() && ele->superCluster()->seed().isNonnull() ?
         ele->deltaEtaSuperClusterTrackAtVtx() - ele->superCluster()->eta() + ele->superCluster()->seed()->eta() : std::numeric_limits<float>::max();
      }
float ElectronIdentifier::isolation(const reco::GsfElectronPtr& ele) {
  if(rho_ < 0 ) {
     throw;
  }
  const reco::GsfElectron::PflowIsolationVariables& pfIso = ele->pfIsolationVariables();
  const float chad = pfIso.sumChargedHadronPt;
  const float nhad = pfIso.sumNeutralHadronEt;
  const float pho = pfIso.sumPhotonEt;
  const float  eA = _effectiveAreas.getEffectiveArea( fabs(ele->superCluster()->eta()) );
  const float iso = chad + std::max(0.0, nhad + pho - rho_*eA);

  // Apply the cut and return the result
  // Scale by pT if the relative isolation is requested but avoid division by 0
  return iso;
}


bool ElectronIdentifier::passID(const reco::GsfElectronPtr& ele,edm::Handle<reco::BeamSpot> beamspot,edm::Handle<reco::ConversionCollection> conversions) {
   if(ID_ == -1) throw;
   unsigned int region = fabs(ele->superCluster()->eta()) < 1.479 ? EleIDEtaBins::BARREL : EleIDEtaBins::BARREL;

   if( ele->full5x5_sigmaIetaIeta()                     > cuts_[EleIDCutNames::SIGMAIETA][ID_][region]) return false;
   if( dEtaInSeed(ele)                                  > cuts_[EleIDCutNames::DETAINSEED][ID_][region]) return false;
   if( std::abs(ele->deltaPhiSuperClusterTrackAtVtx())  > cuts_[EleIDCutNames::DPHIIN][ID_][region]) return false;
   if( ele->hadronicOverEm()                            > cuts_[EleIDCutNames::HOVERE][ID_][region]) return false;
   if( isolation(ele)/ele->pt()                         > cuts_[EleIDCutNames::ISO][ID_][region]) return false;
   if( std::abs(1.0 - ele->eSuperClusterOverP())/ele->ecalEnergy()  > cuts_[EleIDCutNames::ONEOVERE][ID_][region]) return false;
   if( (ele->gsfTrack()->hitPattern().numberOfAllHits(reco::HitPattern::MISSING_INNER_HITS)) > cuts_[EleIDCutNames::MISSINGHITS][ID_][region]) return false;
   if( ConversionTools::hasMatchedConversion(*ele,*conversions,beamspot->position())) return false;

   return true;
}




