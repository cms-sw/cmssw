// $Id: EGEnergyCorrector.cc,v 1.8 2011/12/14 20:16:56 bendavid Exp $

#include <TFile.h>
#include "../interface/EGEnergyCorrector.h"
#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "FWCore/Framework/interface/ESHandle.h" 
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

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
    
    if (fVals) delete [] fVals;
    if (fOwnsForests) {
      if (fReadereb) delete fReadereb;
      if (fReaderebvariance) delete fReaderebvariance;  
      if (fReaderee) delete fReaderee;
      if (fReadereevariance) delete fReadereevariance;    
    }    

    fVals = new Float_t[73];
    
    if (weightsFromDB) { //weights from event setup
      
      edm::ESHandle<GBRForest> readereb;
      edm::ESHandle<GBRForest> readerebvar;
      edm::ESHandle<GBRForest> readeree;
      edm::ESHandle<GBRForest> readereevar;

      iSetup.get<GBRWrapperRcd>().get(std::string(TString::Format("%s_EBCorrection",regweights.c_str())),readereb);
      iSetup.get<GBRWrapperRcd>().get(std::string(TString::Format("%s_EBUncertainty",regweights.c_str())),readerebvar);
      iSetup.get<GBRWrapperRcd>().get(std::string(TString::Format("%s_EECorrection",regweights.c_str())),readeree);
      iSetup.get<GBRWrapperRcd>().get(std::string(TString::Format("%s_EEUncertainty",regweights.c_str())),readereevar);

      fReadereb = readereb.product();
      fReaderebvariance = readerebvar.product();
      fReaderee = readeree.product();
      fReadereevariance = readereevar.product();

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
std::pair<double,double> EGEnergyCorrector::CorrectedEnergyWithError(const Photon &p, const reco::VertexCollection& vtxcol, EcalClusterLazyTools &clustertools, const edm::EventSetup &es) {
  
  const SuperClusterRef s = p.superCluster();
  const CaloClusterPtr b = s->seed(); //seed  basic cluster

  //highest energy basic cluster excluding seed basic cluster
  CaloClusterPtr b2;
  Double_t ebcmax = -99.;
  for (reco::CaloCluster_iterator bit = s->clustersBegin(); bit!=s->clustersEnd(); ++bit) {
    const CaloClusterPtr bc = *bit;
    if (bc->energy() > ebcmax && bc !=b) {
      b2 = bc;
      ebcmax = bc->energy();
    }
  }

  //lowest energy basic cluster excluding seed (for pileup mitigation)
  CaloClusterPtr bclast;
  Double_t ebcmin = 1e6;
  for (reco::CaloCluster_iterator bit = s->clustersBegin(); bit!=s->clustersEnd(); ++bit) {
    const CaloClusterPtr bc = *bit;
    if (bc->energy() < ebcmin && bc !=b) {
      bclast = bc;
      ebcmin = bc->energy();
    }
  }

  //2nd lowest energy basic cluster excluding seed (for pileup mitigation)
  CaloClusterPtr bclast2;
  ebcmin = 1e6;
  for (reco::CaloCluster_iterator bit = s->clustersBegin(); bit!=s->clustersEnd(); ++bit) {
    const CaloClusterPtr bc = *bit;
    if (bc->energy() < ebcmin && bc !=b && bc!=bclast) {
      bclast2 = bc;
      ebcmin = bc->energy();
    }
  }
  
  Bool_t isbarrel =  b->hitsAndFractions().at(0).first.subdetId()==EcalBarrel;
  Bool_t hasbc2 = b2.isNonnull() && b2->energy()>0.;
  Bool_t hasbclast = bclast.isNonnull() && bclast->energy()>0.;
  Bool_t hasbclast2 = bclast2.isNonnull() && bclast2->energy()>0.;
  
  
  if (isbarrel) {
    
    //basic supercluster variables
    fVals[0]  = s->rawEnergy();
    fVals[1]  = p.r9();
    fVals[2]  = s->eta();
    fVals[3]  = s->phi();
    fVals[4]  = p.e5x5()/s->rawEnergy();   
    fVals[5] = p.hadronicOverEm();
    fVals[6] = s->etaWidth();
    fVals[7] = s->phiWidth();
    
    
    //seed basic cluster variables
    double bemax = clustertools.eMax(*b);
    double be2nd = clustertools.e2nd(*b);
    double betop = clustertools.eTop(*b);
    double bebottom = clustertools.eBottom(*b);
    double beleft = clustertools.eLeft(*b);
    double beright = clustertools.eRight(*b);
    
    
    fVals[8] = b->eta()-s->eta();
    fVals[9] = reco::deltaPhi(b->phi(),s->phi());
    fVals[10] = b->energy()/s->rawEnergy();
    fVals[11] = clustertools.e3x3(*b)/b->energy();
    fVals[12] = clustertools.e5x5(*b)/b->energy();
    fVals[13] = sqrt(clustertools.localCovariances(*b)[0]); //sigietaieta
    fVals[14] = sqrt(clustertools.localCovariances(*b)[2]); //sigiphiiphi
    fVals[15] = clustertools.localCovariances(*b)[1];       //sigietaiphi
    fVals[16] = bemax/b->energy();                       //crystal energy ratio gap variables   
    fVals[17] = log(be2nd/bemax);
    fVals[18] = log(betop/bemax);
    fVals[19] = log(bebottom/bemax);
    fVals[20] = log(beleft/bemax);
    fVals[21] = log(beright/bemax);
    fVals[22] = (betop-bebottom)/(betop+bebottom);
    fVals[23] = (beleft-beright)/(beleft+beright);

    
    double bc2emax = hasbc2 ? clustertools.eMax(*b2) : 0.;
    double bc2e2nd = hasbc2 ? clustertools.e2nd(*b2) : 0.;
    double bc2etop = hasbc2 ? clustertools.eTop(*b2) : 0.;
    double bc2ebottom = hasbc2 ? clustertools.eBottom(*b2) : 0.;
    double bc2eleft = hasbc2 ? clustertools.eLeft(*b2) : 0.;
    double bc2eright = hasbc2 ? clustertools.eRight(*b2) : 0.;
    
    fVals[24] = hasbc2 ? (b2->eta()-s->eta()) : 0.;
    fVals[25] = hasbc2 ? reco::deltaPhi(b2->phi(),s->phi()) : 0.;
    fVals[26] = hasbc2 ? b2->energy()/s->rawEnergy() : 0.;
    fVals[27] = hasbc2 ? clustertools.e3x3(*b2)/b2->energy() : 0.;
    fVals[28] = hasbc2 ? clustertools.e5x5(*b2)/b2->energy() : 0.;
    fVals[29] = hasbc2 ? sqrt(clustertools.localCovariances(*b2)[0]) : 0.;
    fVals[30] = hasbc2 ? sqrt(clustertools.localCovariances(*b2)[2]) : 0.;
    fVals[31] = hasbc2 ? clustertools.localCovariances(*b)[1] : 0.;
    fVals[32] = hasbc2 ? bc2emax/b2->energy() : 0.;
    fVals[33] = hasbc2 ? log(bc2e2nd/bc2emax) : 0.;
    fVals[34] = hasbc2 ? log(bc2etop/bc2emax) : 0.;
    fVals[35] = hasbc2 ? log(bc2ebottom/bc2emax) : 0.;
    fVals[36] = hasbc2 ? log(bc2eleft/bc2emax) : 0.;
    fVals[37] = hasbc2 ? log(bc2eright/bc2emax) : 0.;
    fVals[38] = hasbc2 ? (bc2etop-bc2ebottom)/(bc2etop+bc2ebottom) : 0.;
    fVals[39] = hasbc2 ? (bc2eleft-bc2eright)/(bc2eleft+bc2eright) : 0.;

    fVals[40] = hasbclast ? (bclast->eta()-s->eta()) : 0.;
    fVals[41] = hasbclast ? reco::deltaPhi(bclast->phi(),s->phi()) : 0.;
    fVals[42] = hasbclast ? bclast->energy()/s->rawEnergy() : 0.;
    fVals[43] = hasbclast ? clustertools.e3x3(*bclast)/bclast->energy() : 0.;
    fVals[44] = hasbclast ? clustertools.e5x5(*bclast)/bclast->energy() : 0.;
    fVals[45] = hasbclast ? sqrt(clustertools.localCovariances(*bclast)[0]) : 0.;
    fVals[46] = hasbclast ? sqrt(clustertools.localCovariances(*bclast)[2]) : 0.;
    fVals[47] = hasbclast ? clustertools.localCovariances(*bclast)[1] : 0.;    

    fVals[48] = hasbclast2 ? (bclast2->eta()-s->eta()) : 0.;
    fVals[49] = hasbclast2 ? reco::deltaPhi(bclast2->phi(),s->phi()) : 0.;
    fVals[50] = hasbclast2 ? bclast2->energy()/s->rawEnergy() : 0.;
    fVals[51] = hasbclast2 ? clustertools.e3x3(*bclast2)/bclast2->energy() : 0.;
    fVals[52] = hasbclast2 ? clustertools.e5x5(*bclast2)/bclast2->energy() : 0.;
    fVals[53] = hasbclast2 ? sqrt(clustertools.localCovariances(*bclast2)[0]) : 0.;
    fVals[54] = hasbclast2 ? sqrt(clustertools.localCovariances(*bclast2)[2]) : 0.;
    fVals[55] = hasbclast2 ? clustertools.localCovariances(*bclast2)[1] : 0.;    
        
    
    //local coordinates and crystal indices
    
    
    //seed cluster
    float betacry, bphicry, bthetatilt, bphitilt;
    int bieta, biphi;
    _ecalLocal.localCoordsEB(*b,es,betacry,bphicry,bieta,biphi,bthetatilt,bphitilt);
    
    fVals[56] = bieta; //crystal ieta
    fVals[57] = biphi; //crystal iphi
    fVals[58] = bieta%5; //submodule boundary eta symmetry
    fVals[59] = biphi%2; //submodule boundary phi symmetry
    fVals[60] = (TMath::Abs(bieta)<=25)*(bieta%25) + (TMath::Abs(bieta)>25)*((bieta-25*TMath::Abs(bieta)/bieta)%20);  //module boundary eta approximate symmetry
    fVals[61] = biphi%20; //module boundary phi symmetry
    fVals[62] = betacry; //local coordinates with respect to closest crystal center at nominal shower depth
    fVals[63] = bphicry;

    
    //2nd cluster (meaningful gap corrections for converted photons)
    float bc2etacry, bc2phicry, bc2thetatilt, bc2phitilt;
    int bc2ieta, bc2iphi;
    if (hasbc2) _ecalLocal.localCoordsEB(*b2,es,bc2etacry,bc2phicry,bc2ieta,bc2iphi,bc2thetatilt,bc2phitilt);    
    
    fVals[64] = hasbc2 ? bc2ieta : 0.;
    fVals[65] = hasbc2 ? bc2iphi : 0.;
    fVals[66] = hasbc2 ? bc2ieta%5 : 0.;
    fVals[67] = hasbc2 ? bc2iphi%2 : 0.;
    fVals[68] = hasbc2 ? (TMath::Abs(bc2ieta)<=25)*(bc2ieta%25) + (TMath::Abs(bc2ieta)>25)*((bc2ieta-25*TMath::Abs(bc2ieta)/bc2ieta)%20) : 0.;
    fVals[69] = hasbc2 ? bc2iphi%20 : 0.;
    fVals[70] = hasbc2 ? bc2etacry : 0.;
    fVals[71] = hasbc2 ? bc2phicry : 0.;    
    
    fVals[72] = vtxcol.size();
    
  }
  else {
    fVals[0]  = s->rawEnergy();
    fVals[1]  = p.r9();
    fVals[2]  = s->eta();
    fVals[3]  = s->phi();
    fVals[4]  = p.e5x5()/s->rawEnergy();
    fVals[5] = s->etaWidth();
    fVals[6] = s->phiWidth();
    fVals[7] = vtxcol.size();
  }
    
    
  const Double_t varscale = 1.253;
  Double_t den;
  const GBRForest *reader;
  const GBRForest *readervar;
  if (isbarrel) {
    den = s->rawEnergy();
    reader = fReadereb;
    readervar = fReaderebvariance;
  }
  else {
    den = s->rawEnergy() + s->preshowerEnergy();
    reader = fReaderee;
    readervar = fReadereevariance;
  }
  
  Double_t ecor = reader->GetResponse(fVals)*den;
  Double_t ecorerr = readervar->GetResponse(fVals)*den*varscale;
  
  //printf("ecor = %5f, ecorerr = %5f\n",ecor,ecorerr);
  
  return std::pair<double,double>(ecor,ecorerr);
}


//--------------------------------------------------------------------------------------------------
std::pair<double,double> EGEnergyCorrector::CorrectedEnergyWithError(const GsfElectron &e, const reco::VertexCollection& vtxcol, EcalClusterLazyTools &clustertools, const edm::EventSetup &es) {
  
  //apply v2 regression to electrons
  //mostly duplicated from photon function above //TODO, make common underlying function
  
  //protection, this doesn't work properly on non-egamma-seeded electrons
  if (!e.ecalDrivenSeed()) return std::pair<double,double>(0.,0.);
  
  
  const SuperClusterRef s = e.superCluster();
  const CaloClusterPtr b = s->seed();

  CaloClusterPtr b2;
  Double_t ebcmax = -99.;
  for (reco::CaloCluster_iterator bit = s->clustersBegin(); bit!=s->clustersEnd(); ++bit) {
    const CaloClusterPtr bc = *bit;
    if (bc->energy() > ebcmax && bc !=b) {
      b2 = bc;
      ebcmax = bc->energy();
    }
  }

  CaloClusterPtr bclast;
  Double_t ebcmin = 1e6;
  for (reco::CaloCluster_iterator bit = s->clustersBegin(); bit!=s->clustersEnd(); ++bit) {
    const CaloClusterPtr bc = *bit;
    if (bc->energy() < ebcmin && bc !=b) {
      bclast = bc;
      ebcmin = bc->energy();
    }
  }

  CaloClusterPtr bclast2;
  ebcmin = 1e6;
  for (reco::CaloCluster_iterator bit = s->clustersBegin(); bit!=s->clustersEnd(); ++bit) {
    const CaloClusterPtr bc = *bit;
    if (bc->energy() < ebcmin && bc !=b && bc!=bclast) {
      bclast2 = bc;
      ebcmin = bc->energy();
    }
  }
  
  Bool_t isbarrel =  b->hitsAndFractions().at(0).first.subdetId()==EcalBarrel;
  Bool_t hasbc2 = b2.isNonnull() && b2->energy()>0.;
  Bool_t hasbclast = bclast.isNonnull() && bclast->energy()>0.;
  Bool_t hasbclast2 = bclast2.isNonnull() && bclast2->energy()>0.;
  
  
  if (isbarrel) {
    fVals[0]  = s->rawEnergy();
    fVals[1]  = clustertools.e3x3(*b)/s->rawEnergy(); //r9
    fVals[2]  = s->eta();
    fVals[3]  = s->phi();
    fVals[4]  = clustertools.e5x5(*b)/s->rawEnergy();
    fVals[5] = e.hcalOverEcal();
    fVals[6] = s->etaWidth();
    fVals[7] = s->phiWidth();
    
    
    double bemax = clustertools.eMax(*b);
    double be2nd = clustertools.e2nd(*b);
    double betop = clustertools.eTop(*b);
    double bebottom = clustertools.eBottom(*b);
    double beleft = clustertools.eLeft(*b);
    double beright = clustertools.eRight(*b);
    
    
    fVals[8] = b->eta()-s->eta();
    fVals[9] = reco::deltaPhi(b->phi(),s->phi());
    fVals[10] = b->energy()/s->rawEnergy();
    fVals[11] = clustertools.e3x3(*b)/b->energy();
    fVals[12] = clustertools.e5x5(*b)/b->energy();
    fVals[13] = sqrt(clustertools.localCovariances(*b)[0]);
    fVals[14] = sqrt(clustertools.localCovariances(*b)[2]);
    fVals[15] = clustertools.localCovariances(*b)[1];
    fVals[16] = bemax/b->energy();
    fVals[17] = log(be2nd/bemax);
    fVals[18] = log(betop/bemax);
    fVals[19] = log(bebottom/bemax);
    fVals[20] = log(beleft/bemax);
    fVals[21] = log(beright/bemax);
    fVals[22] = (betop-bebottom)/(betop+bebottom);
    fVals[23] = (beleft-beright)/(beleft+beright);

    
    double bc2emax = hasbc2 ? clustertools.eMax(*b2) : 0.;
    double bc2e2nd = hasbc2 ? clustertools.e2nd(*b2) : 0.;
    double bc2etop = hasbc2 ? clustertools.eTop(*b2) : 0.;
    double bc2ebottom = hasbc2 ? clustertools.eBottom(*b2) : 0.;
    double bc2eleft = hasbc2 ? clustertools.eLeft(*b2) : 0.;
    double bc2eright = hasbc2 ? clustertools.eRight(*b2) : 0.;
    
    fVals[24] = hasbc2 ? (b2->eta()-s->eta()) : 0.;
    fVals[25] = hasbc2 ? reco::deltaPhi(b2->phi(),s->phi()) : 0.;
    fVals[26] = hasbc2 ? b2->energy()/s->rawEnergy() : 0.;
    fVals[27] = hasbc2 ? clustertools.e3x3(*b2)/b2->energy() : 0.;
    fVals[28] = hasbc2 ? clustertools.e5x5(*b2)/b2->energy() : 0.;
    fVals[29] = hasbc2 ? sqrt(clustertools.localCovariances(*b2)[0]) : 0.;
    fVals[30] = hasbc2 ? sqrt(clustertools.localCovariances(*b2)[2]) : 0.;
    fVals[31] = hasbc2 ? clustertools.localCovariances(*b)[1] : 0.;
    fVals[32] = hasbc2 ? bc2emax/b2->energy() : 0.;
    fVals[33] = hasbc2 ? log(bc2e2nd/bc2emax) : 0.;
    fVals[34] = hasbc2 ? log(bc2etop/bc2emax) : 0.;
    fVals[35] = hasbc2 ? log(bc2ebottom/bc2emax) : 0.;
    fVals[36] = hasbc2 ? log(bc2eleft/bc2emax) : 0.;
    fVals[37] = hasbc2 ? log(bc2eright/bc2emax) : 0.;
    fVals[38] = hasbc2 ? (bc2etop-bc2ebottom)/(bc2etop+bc2ebottom) : 0.;
    fVals[39] = hasbc2 ? (bc2eleft-bc2eright)/(bc2eleft+bc2eright) : 0.;

    fVals[40] = hasbclast ? (bclast->eta()-s->eta()) : 0.;
    fVals[41] = hasbclast ? reco::deltaPhi(bclast->phi(),s->phi()) : 0.;
    fVals[42] = hasbclast ? bclast->energy()/s->rawEnergy() : 0.;
    fVals[43] = hasbclast ? clustertools.e3x3(*bclast)/bclast->energy() : 0.;
    fVals[44] = hasbclast ? clustertools.e5x5(*bclast)/bclast->energy() : 0.;
    fVals[45] = hasbclast ? sqrt(clustertools.localCovariances(*bclast)[0]) : 0.;
    fVals[46] = hasbclast ? sqrt(clustertools.localCovariances(*bclast)[2]) : 0.;
    fVals[47] = hasbclast ? clustertools.localCovariances(*bclast)[1] : 0.;    

    fVals[48] = hasbclast2 ? (bclast2->eta()-s->eta()) : 0.;
    fVals[49] = hasbclast2 ? reco::deltaPhi(bclast2->phi(),s->phi()) : 0.;
    fVals[50] = hasbclast2 ? bclast2->energy()/s->rawEnergy() : 0.;
    fVals[51] = hasbclast2 ? clustertools.e3x3(*bclast2)/bclast2->energy() : 0.;
    fVals[52] = hasbclast2 ? clustertools.e5x5(*bclast2)/bclast2->energy() : 0.;
    fVals[53] = hasbclast2 ? sqrt(clustertools.localCovariances(*bclast2)[0]) : 0.;
    fVals[54] = hasbclast2 ? sqrt(clustertools.localCovariances(*bclast2)[2]) : 0.;
    fVals[55] = hasbclast2 ? clustertools.localCovariances(*bclast2)[1] : 0.;    
        
    
    float betacry, bphicry, bthetatilt, bphitilt;
    int bieta, biphi;
    _ecalLocal.localCoordsEB(*b,es,betacry,bphicry,bieta,biphi,bthetatilt,bphitilt);
    
    fVals[56] = bieta;
    fVals[57] = biphi;
    fVals[58] = bieta%5;
    fVals[59] = biphi%2;
    fVals[60] = (TMath::Abs(bieta)<=25)*(bieta%25) + (TMath::Abs(bieta)>25)*((bieta-25*TMath::Abs(bieta)/bieta)%20);
    fVals[61] = biphi%20;
    fVals[62] = betacry;
    fVals[63] = bphicry;

    float bc2etacry, bc2phicry, bc2thetatilt, bc2phitilt;
    int bc2ieta, bc2iphi;
    if (hasbc2) _ecalLocal.localCoordsEB(*b2,es,bc2etacry,bc2phicry,bc2ieta,bc2iphi,bc2thetatilt,bc2phitilt);    
    
    fVals[64] = hasbc2 ? bc2ieta : 0.;
    fVals[65] = hasbc2 ? bc2iphi : 0.;
    fVals[66] = hasbc2 ? bc2ieta%5 : 0.;
    fVals[67] = hasbc2 ? bc2iphi%2 : 0.;
    fVals[68] = hasbc2 ? (TMath::Abs(bc2ieta)<=25)*(bc2ieta%25) + (TMath::Abs(bc2ieta)>25)*((bc2ieta-25*TMath::Abs(bc2ieta)/bc2ieta)%20) : 0.;
    fVals[69] = hasbc2 ? bc2iphi%20 : 0.;
    fVals[70] = hasbc2 ? bc2etacry : 0.;
    fVals[71] = hasbc2 ? bc2phicry : 0.;    
    
    fVals[72] = vtxcol.size();
    
  }
  else {
    fVals[0]  = s->rawEnergy();
    fVals[1]  = clustertools.e3x3(*b)/s->rawEnergy(); //r9
    fVals[2]  = s->eta();
    fVals[3]  = s->phi();
    fVals[4]  = clustertools.e5x5(*b)/s->rawEnergy();
    fVals[5] = s->etaWidth();
    fVals[6] = s->phiWidth();
    fVals[7] = vtxcol.size();
  }
    
    
  const Double_t varscale = 1.253;
  Double_t den;
  const GBRForest *reader;
  const GBRForest *readervar;
  if (isbarrel) {
    den = s->rawEnergy();
    reader = fReadereb;
    readervar = fReaderebvariance;
  }
  else {
    den = s->rawEnergy() + s->preshowerEnergy();
    reader = fReaderee;
    readervar = fReadereevariance;
  }
  
  Double_t ecor = reader->GetResponse(fVals)*den;
  Double_t ecorerr = readervar->GetResponse(fVals)*den*varscale;
  
  //printf("ecor = %5f, ecorerr = %5f\n",ecor,ecorerr);
  
  return std::pair<double,double>(ecor,ecorerr);

}