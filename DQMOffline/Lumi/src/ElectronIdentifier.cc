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

#include <boost/foreach.hpp>
#include <TLorentzVector.h>
#include <TMath.h>
#include <algorithm>

ElectronIdentifier::ElectronIdentifier (const edm::ParameterSet& c) :
   _effectiveAreas( (c.getParameter<edm::FileInPath>("effAreasConfigFile")).fullPath())

{
   rho_ = -1;
   ID_ = "NULL";
   cuts_["SIGMAIETA"]["VETO"]  ["BARREL"] = 0.0115;
   cuts_["SIGMAIETA"]["LOOSE"] ["BARREL"] = 0.011;
   cuts_["SIGMAIETA"]["MEDIUM"]["BARREL"] = 0.00998;
   cuts_["SIGMAIETA"]["TIGHT"] ["BARREL"] = 0.00998;

   cuts_["SIGMAIETA"]["VETO"]  ["ENDCAP"] = 0.037;
   cuts_["SIGMAIETA"]["LOOSE"] ["ENDCAP"] = 0.0314;
   cuts_["SIGMAIETA"]["MEDIUM"]["ENDCAP"] = 0.0298;
   cuts_["SIGMAIETA"]["TIGHT"] ["ENDCAP"] = 0.0292;

   cuts_["DETAINSEED"]["VETO"]  ["BARREL"] = 0.00749;
   cuts_["DETAINSEED"]["LOOSE"] ["BARREL"] = 0.00477;
   cuts_["DETAINSEED"]["MEDIUM"]["BARREL"] = 0.00311;
   cuts_["DETAINSEED"]["TIGHT"] ["BARREL"] = 0.00308;

   cuts_["DETAINSEED"]["VETO"]  ["ENDCAP"] = 0.00895;
   cuts_["DETAINSEED"]["LOOSE"] ["ENDCAP"] = 0.00868;
   cuts_["DETAINSEED"]["MEDIUM"]["ENDCAP"] = 0.00609;
   cuts_["DETAINSEED"]["TIGHT"] ["ENDCAP"] = 0.00605;

   cuts_["DPHIIN"]["VETO"]  ["BARREL"] = 0.228;
   cuts_["DPHIIN"]["LOOSE"] ["BARREL"] = 	0.222;
   cuts_["DPHIIN"]["MEDIUM"]["BARREL"] = 	0.103;
   cuts_["DPHIIN"]["TIGHT"] ["BARREL"] = 	0.0816;

   cuts_["DPHIIN"]["VETO"]  ["ENDCAP"] = 0.213;
   cuts_["DPHIIN"]["LOOSE"] ["ENDCAP"] = 	0.213;
   cuts_["DPHIIN"]["MEDIUM"]["ENDCAP"] = 	0.045;
   cuts_["DPHIIN"]["TIGHT"] ["ENDCAP"] = 	0.0394;

   cuts_["HOVERE"]["VETO"]  ["BARREL"] = 0.356;
   cuts_["HOVERE"]["LOOSE"] ["BARREL"] = 	0.298;
   cuts_["HOVERE"]["MEDIUM"]["BARREL"] = 0.253;
   cuts_["HOVERE"]["TIGHT"] ["BARREL"] = 0.0414;

   cuts_["HOVERE"]["VETO"]  ["ENDCAP"] = 0.211;
   cuts_["HOVERE"]["LOOSE"] ["ENDCAP"] = 0.101;
   cuts_["HOVERE"]["MEDIUM"]["ENDCAP"] = 0.0878;
   cuts_["HOVERE"]["TIGHT"] ["ENDCAP"] = 0.0641;

   cuts_["ISO"]["VETO"]  ["BARREL"] = 0.175;
   cuts_["ISO"]["LOOSE"] ["BARREL"] = 0.0994;
   cuts_["ISO"]["MEDIUM"]["BARREL"] = 	0.0695;
   cuts_["ISO"]["TIGHT"] ["BARREL"] = 	0.0588;

   cuts_["ISO"]["VETO"]  ["ENDCAP"] = 0.159;
   cuts_["ISO"]["LOOSE"] ["ENDCAP"] = 	0.107;
   cuts_["ISO"]["MEDIUM"]["ENDCAP"] = 	0.0821;
   cuts_["ISO"]["TIGHT"] ["ENDCAP"] = 	0.0571;

   cuts_["1OVERE"]["VETO"]  ["BARREL"] = 0.299;
   cuts_["1OVERE"]["LOOSE"] ["BARREL"] = 0.241;
   cuts_["1OVERE"]["MEDIUM"]["BARREL"] = 0.134;
   cuts_["1OVERE"]["TIGHT"] ["BARREL"] = 0.0129;

   cuts_["1OVERE"]["VETO"]  ["ENDCAP"] = 0.15;
   cuts_["1OVERE"]["LOOSE"] ["ENDCAP"] = 0.14;
   cuts_["1OVERE"]["MEDIUM"]["ENDCAP"] = 0.13;
   cuts_["1OVERE"]["TIGHT"] ["ENDCAP"] = 0.0129;

   cuts_["MISSINGHITS"]["VETO"]  ["BARREL"] = 2;
   cuts_["MISSINGHITS"]["LOOSE"] ["BARREL"] = 1;
   cuts_["MISSINGHITS"]["MEDIUM"]["BARREL"] = 1;
   cuts_["MISSINGHITS"]["TIGHT"] ["BARREL"] = 1;

   cuts_["MISSINGHITS"]["VETO"]  ["ENDCAP"] = 3;
   cuts_["MISSINGHITS"]["LOOSE"] ["ENDCAP"] = 1;
   cuts_["MISSINGHITS"]["MEDIUM"]["ENDCAP"] = 1;
   cuts_["MISSINGHITS"]["TIGHT"] ["ENDCAP"] = 1;

   cuts_["CONVERSION"]["VETO"]  ["BARREL"] = 1;
   cuts_["CONVERSION"]["LOOSE"] ["BARREL"] = 1;
   cuts_["CONVERSION"]["MEDIUM"]["BARREL"] = 1;
   cuts_["CONVERSION"]["TIGHT"] ["BARREL"] = 1;

   cuts_["CONVERSION"]["VETO"]  ["ENDCAP"] = 1;
   cuts_["CONVERSION"]["LOOSE"] ["ENDCAP"] = 1;
   cuts_["CONVERSION"]["MEDIUM"]["ENDCAP"] = 1;
   cuts_["CONVERSION"]["TIGHT"] ["ENDCAP"] = 1;
}

void ElectronIdentifier::setRho(double rho) {
   if(rho >= 0) {
      rho_ = rho;
   } else {
      throw;
   }
}
void ElectronIdentifier::setID(std::string ID) {
   bool is_available = true;
   for (auto const cutmap : cuts_) {
      bool tmp = false;
      for( auto const item : cutmap.second ){
            tmp |= (item.first == ID);
      }
      is_available &= tmp;
   }
   if(is_available){
      ID_ = ID;
   } else {
      throw;
   }
}
void ElectronIdentifier::setBeamspot(edm::Handle<reco::BeamSpot> beamspot) {
   beamspot_ = beamspot;
}
void ElectronIdentifier::setConversions(edm::Handle<reco::ConversionCollection> conversions) {
   conversions_ = conversions;
}
void ElectronIdentifier::loadEvent(const edm::Event& iEvent){
   //~ iEvent.getByToken(fRhoToken, rho);
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


bool ElectronIdentifier::passID(const reco::GsfElectronPtr& ele) {
   if(ID_ == "NULL") throw;
   std::string region = fabs(ele->superCluster()->eta()) < 1.479 ? "BARREL" : "ENDCAP";

   bool pass = true;

   std::vector<bool> passes;
   passes.push_back( ele->full5x5_sigmaIetaIeta()                     < cuts_["SIGMAIETA"][ID_][region]);
   passes.push_back( dEtaInSeed(ele)                                  < cuts_["DETAINSEED"][ID_][region]);
   passes.push_back( std::abs(ele->deltaPhiSuperClusterTrackAtVtx())  < cuts_["DPHIIN"][ID_][region]);
   passes.push_back( ele->hadronicOverEm()                            < cuts_["HOVERE"][ID_][region]);
   passes.push_back( isolation(ele)/ele->pt()                         < cuts_["ISO"][ID_][region]);
   passes.push_back( std::abs(1.0 - ele->eSuperClusterOverP())/ele->ecalEnergy()  < cuts_["1OVERE"][ID_][region]);
   passes.push_back( (ele->gsfTrack()->hitPattern().numberOfAllHits(reco::HitPattern::MISSING_INNER_HITS)) <= cuts_["MISSINGHITS"][ID_][region]);
   passes.push_back( !ConversionTools::hasMatchedConversion(*ele,conversions_,beamspot_->position()));

   for (auto const p:passes) {
      pass &= p;
   }
   return pass;
}




