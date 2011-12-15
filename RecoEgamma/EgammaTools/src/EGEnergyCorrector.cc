// $Id: EGEnergyCorrector.cc,v 1.1 2011/11/01 16:16:40 bendavid Exp $

#include <TFile.h>
#include "../interface/EGEnergyCorrector.h"
#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include "CondFormats/EgammaObjects/interface/GBRWrapper.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "FWCore/Framework/interface/ESHandle.h" 
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"


using namespace reco;

//--------------------------------------------------------------------------------------------------
EGEnergyCorrector::EGEnergyCorrector() :
fReadereb(0),
fReaderebvariance(0),
fReaderee(0),
fReadereevariance(0),
fIsInitialized(kFALSE),
fOwnsForests(kFALSE),
fVals(0)
{
  // Constructor.
}


//--------------------------------------------------------------------------------------------------
EGEnergyCorrector::~EGEnergyCorrector()
{
  
  if (fVals) delete [] fVals;

  if (fOwnsForests) {
    if (fReadereb) delete fReadereb;
    if (fReaderebvariance) delete fReaderebvariance;  
    if (fReaderee) delete fReaderee;
    if (fReadereevariance) delete fReadereevariance;
  }

}

//--------------------------------------------------------------------------------------------------
void EGEnergyCorrector::Initialize(const edm::EventSetup &iSetup, std::string regweights, bool weightsFromDB) {
    fIsInitialized = kTRUE;
    
    PhotonFix::initialiseGeometry(iSetup);

    if (fVals) delete [] fVals;
    if (fOwnsForests) {
      if (fReadereb) delete fReadereb;
      if (fReaderebvariance) delete fReaderebvariance;  
      if (fReaderee) delete fReaderee;
      if (fReadereevariance) delete fReadereevariance;    
    }    

    fVals = new Float_t[18];
    
    if (weightsFromDB) { //weights from event setup
      
      edm::ESHandle<GBRWrapper> readereb;
      edm::ESHandle<GBRWrapper> readerebvar;
      edm::ESHandle<GBRWrapper> readeree;
      edm::ESHandle<GBRWrapper> readereevar;

      iSetup.get<GBRWrapperRcd>().get(std::string(TString::Format("%s_EBCorrection",regweights.c_str())),readereb);
      iSetup.get<GBRWrapperRcd>().get(std::string(TString::Format("%s_EBUncertainty",regweights.c_str())),readerebvar);
      iSetup.get<GBRWrapperRcd>().get(std::string(TString::Format("%s_EECorrection",regweights.c_str())),readeree);
      iSetup.get<GBRWrapperRcd>().get(std::string(TString::Format("%s_EEUncertainty",regweights.c_str())),readereevar);

      fReadereb = &readereb->GetForest();
      fReaderebvariance = &readerebvar->GetForest();
      fReaderee = &readeree->GetForest();
      fReadereevariance = &readereevar->GetForest();

    }
    else { //weights from root file
      fOwnsForests = kTRUE;

      TFile *fgbr = new TFile(regweights.c_str(),"READ");
      fReadereb = (GBRForest*)fgbr->Get("EBCorrection");
      fReaderebvariance = (GBRForest*)fgbr->Get("EBUncertainty");  
      fReaderee = (GBRForest*)fgbr->Get("EECorrection");
      fReadereevariance = (GBRForest*)fgbr->Get("EEUncertainty");      
      fgbr->Close();
    }

}

//--------------------------------------------------------------------------------------------------
std::pair<double,double> EGEnergyCorrector::CorrectedEnergyWithError(const Photon &p) {
  
  const SuperCluster &s = *p.superCluster();

  PhotonFix phfix(s.eta(),s.phi()); 
  
  Bool_t isbarrel = (std::abs(s.eta())<1.48);

  if (isbarrel) {
    fVals[0]  = s.rawEnergy();
    fVals[1]  = p.r9();
    fVals[2]  = s.eta();
    fVals[3]  = s.phi();
    fVals[4]  = p.e5x5()/s.rawEnergy();
    fVals[5]  = phfix.etaC();
    fVals[6]  = phfix.etaS();
    fVals[7]  = phfix.etaM();
    fVals[8]  = phfix.phiC();
    fVals[9]  = phfix.phiS();
    fVals[10] = phfix.phiM();    
    fVals[11] = p.hadronicOverEm();
    fVals[12] = s.etaWidth();
    fVals[13] = s.phiWidth();
    fVals[14] = p.sigmaIetaIeta();
  }
  else {
    fVals[0]  = s.rawEnergy();
    fVals[1]  = p.r9();
    fVals[2]  = s.eta();
    fVals[3]  = s.phi();
    fVals[4]  = p.e5x5()/s.rawEnergy();
    fVals[5]  = s.preshowerEnergy()/s.rawEnergy();
    fVals[6]  = phfix.xZ();
    fVals[7]  = phfix.xC();
    fVals[8]  = phfix.xS();
    fVals[9]  = phfix.xM();
    fVals[10] = phfix.yZ();
    fVals[11] = phfix.yC();
    fVals[12] = phfix.yS();
    fVals[13] = phfix.yM();
    fVals[14] = p.hadronicOverEm();
    fVals[15] = s.etaWidth();
    fVals[16] = s.phiWidth();
    fVals[17] = p.sigmaIetaIeta();    
  }
    
  const Double_t varscale = 1.253;
  Double_t den;
  const GBRForest *reader;
  const GBRForest *readervar;
  if (isbarrel) {
    den = s.rawEnergy();
    reader = fReadereb;
    readervar = fReaderebvariance;
  }
  else {
    den = s.rawEnergy() + s.preshowerEnergy();
    reader = fReaderee;
    readervar = fReadereevariance;
  }
  
  Double_t ecor = reader->GetResponse(fVals)*den;
  Double_t ecorerr = readervar->GetResponse(fVals)*den*varscale;
  
  return std::pair<double,double>(ecor,ecorerr);
}

//--------------------------------------------------------------------------------------------------
std::pair<double,double> EGEnergyCorrector::CorrectedEnergyWithError(const GsfElectron &e, EcalClusterLazyTools &clustertools) {
  
  const SuperCluster &s = *e.superCluster();
  const BasicCluster &b = *s.seed();
  
  PhotonFix phfix(s.eta(),s.phi()); 
  
  Bool_t isbarrel = (std::abs(s.eta())<1.48);

  if (isbarrel) {
    fVals[0]  = s.rawEnergy();
    fVals[1]  = clustertools.e3x3(b)/s.rawEnergy(); //r9
    fVals[2]  = s.eta();
    fVals[3]  = s.phi();
    fVals[4]  = clustertools.e5x5(b)/s.rawEnergy();
    fVals[5]  = phfix.etaC();
    fVals[6]  = phfix.etaS();
    fVals[7]  = phfix.etaM();
    fVals[8]  = phfix.phiC();
    fVals[9]  = phfix.phiS();
    fVals[10] = phfix.phiM();    
    fVals[11] = e.hcalOverEcal();
    fVals[12] = s.etaWidth();
    fVals[13] = s.phiWidth();
    fVals[14] = e.sigmaIetaIeta();
  }
  else {
    fVals[0]  = s.rawEnergy();
    fVals[1]  = clustertools.e3x3(b)/s.rawEnergy(); //r9
    fVals[2]  = s.eta();
    fVals[3]  = s.phi();
    fVals[4]  = clustertools.e5x5(b)/s.rawEnergy();
    fVals[5]  = s.preshowerEnergy()/s.rawEnergy();
    fVals[6]  = phfix.xZ();
    fVals[7]  = phfix.xC();
    fVals[8]  = phfix.xS();
    fVals[9]  = phfix.xM();
    fVals[10] = phfix.yZ();
    fVals[11] = phfix.yC();
    fVals[12] = phfix.yS();
    fVals[13] = phfix.yM();
    fVals[14] = e.hcalOverEcal();
    fVals[15] = s.etaWidth();
    fVals[16] = s.phiWidth();
    fVals[17] = e.sigmaIetaIeta();    
  }
    
  const Double_t varscale = 1.253;
  Double_t den;
  const GBRForest *reader;
  const GBRForest *readervar;
  if (isbarrel) {
    den = s.rawEnergy();
    reader = fReadereb;
    readervar = fReaderebvariance;
  }
  else {
    den = s.rawEnergy() + s.preshowerEnergy();
    reader = fReaderee;
    readervar = fReadereevariance;
  }
  
  Double_t ecor = reader->GetResponse(fVals)*den;
  Double_t ecorerr = readervar->GetResponse(fVals)*den*varscale;
  
  return std::pair<double,double>(ecor,ecorerr);
}