#include <TFile.h>
#include "RecoEgamma/EgammaTools/interface/EGEnergyCorrector.h"
#include "CondFormats/GBRForest/interface/GBRForest.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "RecoEgamma/EgammaTools/interface/EcalClusterLocal.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

using namespace reco;

//--------------------------------------------------------------------------------------------------
EGEnergyCorrector::~EGEnergyCorrector() {
  if (fOwnsForests) {
    if (fReadereb)
      delete fReadereb;
    if (fReaderebvariance)
      delete fReaderebvariance;
    if (fReaderee)
      delete fReaderee;
    if (fReadereevariance)
      delete fReadereevariance;
  }
}

//--------------------------------------------------------------------------------------------------
void EGEnergyCorrector::Initialize(const edm::EventSetup &iSetup, std::string regweights, bool weightsFromDB) {
  fIsInitialized = true;

  if (fOwnsForests) {
    if (fReadereb)
      delete fReadereb;
    if (fReaderebvariance)
      delete fReaderebvariance;
    if (fReaderee)
      delete fReaderee;
    if (fReadereevariance)
      delete fReadereevariance;
  }

  fVals.fill(0.0f);

  if (weightsFromDB) {  //weights from event setup

    edm::ESHandle<GBRForest> readereb;
    edm::ESHandle<GBRForest> readerebvar;
    edm::ESHandle<GBRForest> readeree;
    edm::ESHandle<GBRForest> readereevar;

    iSetup.get<GBRWrapperRcd>().get(regweights + "_EBCorrection", readereb);
    iSetup.get<GBRWrapperRcd>().get(regweights + "_EBUncertainty", readerebvar);
    iSetup.get<GBRWrapperRcd>().get(regweights + "_EECorrection", readeree);
    iSetup.get<GBRWrapperRcd>().get(regweights + "_EEUncertainty", readereevar);

    fReadereb = readereb.product();
    fReaderebvariance = readerebvar.product();
    fReaderee = readeree.product();
    fReadereevariance = readereevar.product();

  } else {  //weights from root file
    fOwnsForests = true;

    TFile *fgbr = TFile::Open(regweights.c_str(), "READ");
    fReadereb = (GBRForest *)fgbr->Get("EBCorrection");
    fReaderebvariance = (GBRForest *)fgbr->Get("EBUncertainty");
    fReaderee = (GBRForest *)fgbr->Get("EECorrection");
    fReadereevariance = (GBRForest *)fgbr->Get("EEUncertainty");
    fgbr->Close();
  }
}

//--------------------------------------------------------------------------------------------------
std::pair<double, double> EGEnergyCorrector::CorrectedEnergyWithError(const Photon &p,
                                                                      const reco::VertexCollection &vtxcol,
                                                                      EcalClusterLazyTools &clustertools,
                                                                      CaloGeometry const &caloGeometry) {
  const SuperClusterRef s = p.superCluster();
  const CaloClusterPtr b = s->seed();  //seed  basic cluster

  //highest energy basic cluster excluding seed basic cluster
  CaloClusterPtr b2;
  Double_t ebcmax = -99.;
  for (reco::CaloCluster_iterator bit = s->clustersBegin(); bit != s->clustersEnd(); ++bit) {
    const CaloClusterPtr bc = *bit;
    if (bc->energy() > ebcmax && bc != b) {
      b2 = bc;
      ebcmax = bc->energy();
    }
  }

  //lowest energy basic cluster excluding seed (for pileup mitigation)
  CaloClusterPtr bclast;
  Double_t ebcmin = 1e6;
  for (reco::CaloCluster_iterator bit = s->clustersBegin(); bit != s->clustersEnd(); ++bit) {
    const CaloClusterPtr bc = *bit;
    if (bc->energy() < ebcmin && bc != b) {
      bclast = bc;
      ebcmin = bc->energy();
    }
  }

  //2nd lowest energy basic cluster excluding seed (for pileup mitigation)
  CaloClusterPtr bclast2;
  ebcmin = 1e6;
  for (reco::CaloCluster_iterator bit = s->clustersBegin(); bit != s->clustersEnd(); ++bit) {
    const CaloClusterPtr bc = *bit;
    if (bc->energy() < ebcmin && bc != b && bc != bclast) {
      bclast2 = bc;
      ebcmin = bc->energy();
    }
  }

  Bool_t isbarrel = b->hitsAndFractions().at(0).first.subdetId() == EcalBarrel;
  Bool_t hasbc2 = b2.isNonnull() && b2->energy() > 0.;
  Bool_t hasbclast = bclast.isNonnull() && bclast->energy() > 0.;
  Bool_t hasbclast2 = bclast2.isNonnull() && bclast2->energy() > 0.;

  if (isbarrel) {
    //basic supercluster variables
    fVals[0] = s->rawEnergy();
    fVals[1] = p.r9();
    fVals[2] = s->eta();
    fVals[3] = s->phi();
    fVals[4] = p.e5x5() / s->rawEnergy();
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

    fVals[8] = b->eta() - s->eta();
    fVals[9] = reco::deltaPhi(b->phi(), s->phi());
    fVals[10] = b->energy() / s->rawEnergy();
    fVals[11] = clustertools.e3x3(*b) / b->energy();
    fVals[12] = clustertools.e5x5(*b) / b->energy();
    fVals[13] = sqrt(clustertools.localCovariances(*b)[0]);  //sigietaieta
    fVals[14] = sqrt(clustertools.localCovariances(*b)[2]);  //sigiphiiphi
    fVals[15] = clustertools.localCovariances(*b)[1];        //sigietaiphi
    fVals[16] = bemax / b->energy();                         //crystal energy ratio gap variables
    fVals[17] = log(be2nd / bemax);
    fVals[18] = log(betop / bemax);
    fVals[19] = log(bebottom / bemax);
    fVals[20] = log(beleft / bemax);
    fVals[21] = log(beright / bemax);
    fVals[22] = (betop - bebottom) / (betop + bebottom);
    fVals[23] = (beleft - beright) / (beleft + beright);

    double bc2emax = hasbc2 ? clustertools.eMax(*b2) : 0.;
    double bc2e2nd = hasbc2 ? clustertools.e2nd(*b2) : 0.;
    double bc2etop = hasbc2 ? clustertools.eTop(*b2) : 0.;
    double bc2ebottom = hasbc2 ? clustertools.eBottom(*b2) : 0.;
    double bc2eleft = hasbc2 ? clustertools.eLeft(*b2) : 0.;
    double bc2eright = hasbc2 ? clustertools.eRight(*b2) : 0.;

    fVals[24] = hasbc2 ? (b2->eta() - s->eta()) : 0.;
    fVals[25] = hasbc2 ? reco::deltaPhi(b2->phi(), s->phi()) : 0.;
    fVals[26] = hasbc2 ? b2->energy() / s->rawEnergy() : 0.;
    fVals[27] = hasbc2 ? clustertools.e3x3(*b2) / b2->energy() : 0.;
    fVals[28] = hasbc2 ? clustertools.e5x5(*b2) / b2->energy() : 0.;
    fVals[29] = hasbc2 ? sqrt(clustertools.localCovariances(*b2)[0]) : 0.;
    fVals[30] = hasbc2 ? sqrt(clustertools.localCovariances(*b2)[2]) : 0.;
    fVals[31] = hasbc2 ? clustertools.localCovariances(*b)[1] : 0.;
    fVals[32] = hasbc2 ? bc2emax / b2->energy() : 0.;
    fVals[33] = hasbc2 ? log(bc2e2nd / bc2emax) : 0.;
    fVals[34] = hasbc2 ? log(bc2etop / bc2emax) : 0.;
    fVals[35] = hasbc2 ? log(bc2ebottom / bc2emax) : 0.;
    fVals[36] = hasbc2 ? log(bc2eleft / bc2emax) : 0.;
    fVals[37] = hasbc2 ? log(bc2eright / bc2emax) : 0.;
    fVals[38] = hasbc2 ? (bc2etop - bc2ebottom) / (bc2etop + bc2ebottom) : 0.;
    fVals[39] = hasbc2 ? (bc2eleft - bc2eright) / (bc2eleft + bc2eright) : 0.;

    fVals[40] = hasbclast ? (bclast->eta() - s->eta()) : 0.;
    fVals[41] = hasbclast ? reco::deltaPhi(bclast->phi(), s->phi()) : 0.;
    fVals[42] = hasbclast ? bclast->energy() / s->rawEnergy() : 0.;
    fVals[43] = hasbclast ? clustertools.e3x3(*bclast) / bclast->energy() : 0.;
    fVals[44] = hasbclast ? clustertools.e5x5(*bclast) / bclast->energy() : 0.;
    fVals[45] = hasbclast ? sqrt(clustertools.localCovariances(*bclast)[0]) : 0.;
    fVals[46] = hasbclast ? sqrt(clustertools.localCovariances(*bclast)[2]) : 0.;
    fVals[47] = hasbclast ? clustertools.localCovariances(*bclast)[1] : 0.;

    fVals[48] = hasbclast2 ? (bclast2->eta() - s->eta()) : 0.;
    fVals[49] = hasbclast2 ? reco::deltaPhi(bclast2->phi(), s->phi()) : 0.;
    fVals[50] = hasbclast2 ? bclast2->energy() / s->rawEnergy() : 0.;
    fVals[51] = hasbclast2 ? clustertools.e3x3(*bclast2) / bclast2->energy() : 0.;
    fVals[52] = hasbclast2 ? clustertools.e5x5(*bclast2) / bclast2->energy() : 0.;
    fVals[53] = hasbclast2 ? sqrt(clustertools.localCovariances(*bclast2)[0]) : 0.;
    fVals[54] = hasbclast2 ? sqrt(clustertools.localCovariances(*bclast2)[2]) : 0.;
    fVals[55] = hasbclast2 ? clustertools.localCovariances(*bclast2)[1] : 0.;

    //local coordinates and crystal indices

    //seed cluster
    float betacry, bphicry, bthetatilt, bphitilt;
    int bieta, biphi;
    egammaTools::localEcalClusterCoordsEB(*b, caloGeometry, betacry, bphicry, bieta, biphi, bthetatilt, bphitilt);

    fVals[56] = bieta;      //crystal ieta
    fVals[57] = biphi;      //crystal iphi
    fVals[58] = bieta % 5;  //submodule boundary eta symmetry
    fVals[59] = biphi % 2;  //submodule boundary phi symmetry
    fVals[60] = (TMath::Abs(bieta) <= 25) * (bieta % 25) +
                (TMath::Abs(bieta) > 25) *
                    ((bieta - 25 * TMath::Abs(bieta) / bieta) % 20);  //module boundary eta approximate symmetry
    fVals[61] = biphi % 20;                                           //module boundary phi symmetry
    fVals[62] = betacry;  //local coordinates with respect to closest crystal center at nominal shower depth
    fVals[63] = bphicry;

    //2nd cluster (meaningful gap corrections for converted photons)
    float bc2etacry, bc2phicry, bc2thetatilt, bc2phitilt;
    int bc2ieta, bc2iphi;
    if (hasbc2)
      egammaTools::localEcalClusterCoordsEB(
          *b2, caloGeometry, bc2etacry, bc2phicry, bc2ieta, bc2iphi, bc2thetatilt, bc2phitilt);

    fVals[64] = hasbc2 ? bc2ieta : 0.;
    fVals[65] = hasbc2 ? bc2iphi : 0.;
    fVals[66] = hasbc2 ? bc2ieta % 5 : 0.;
    fVals[67] = hasbc2 ? bc2iphi % 2 : 0.;
    fVals[68] = hasbc2 ? (TMath::Abs(bc2ieta) <= 25) * (bc2ieta % 25) +
                             (TMath::Abs(bc2ieta) > 25) * ((bc2ieta - 25 * TMath::Abs(bc2ieta) / bc2ieta) % 20)
                       : 0.;
    fVals[69] = hasbc2 ? bc2iphi % 20 : 0.;
    fVals[70] = hasbc2 ? bc2etacry : 0.;
    fVals[71] = hasbc2 ? bc2phicry : 0.;

    fVals[72] = vtxcol.size();

  } else {
    fVals[0] = s->rawEnergy();
    fVals[1] = p.r9();
    fVals[2] = s->eta();
    fVals[3] = s->phi();
    fVals[4] = p.e5x5() / s->rawEnergy();
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
  } else {
    den = s->rawEnergy() + s->preshowerEnergy();
    reader = fReaderee;
    readervar = fReadereevariance;
  }

  Double_t ecor = reader->GetResponse(fVals.data()) * den;
  Double_t ecorerr = readervar->GetResponse(fVals.data()) * den * varscale;

  //printf("ecor = %5f, ecorerr = %5f\n",ecor,ecorerr);

  return {ecor, ecorerr};
}

//--------------------------------------------------------------------------------------------------
std::pair<double, double> EGEnergyCorrector::CorrectedEnergyWithErrorV3(const Photon &p,
                                                                        const reco::VertexCollection &vtxcol,
                                                                        double rho,
                                                                        EcalClusterLazyTools &clustertools,
                                                                        CaloGeometry const &caloGeometry,
                                                                        bool applyRescale) {
  const SuperClusterRef s = p.superCluster();
  const CaloClusterPtr b = s->seed();  //seed  basic cluster

  Bool_t isbarrel = b->hitsAndFractions().at(0).first.subdetId() == EcalBarrel;

  //basic supercluster variables
  fVals[0] = s->rawEnergy();
  fVals[1] = s->eta();
  fVals[2] = s->phi();
  fVals[3] = p.r9();
  fVals[4] = p.e5x5() / s->rawEnergy();
  fVals[5] = s->etaWidth();
  fVals[6] = s->phiWidth();
  fVals[7] = s->clustersSize();
  fVals[8] = p.hadTowOverEm();
  fVals[9] = rho;
  fVals[10] = vtxcol.size();

  //seed basic cluster variables
  double bemax = clustertools.eMax(*b);
  double be2nd = clustertools.e2nd(*b);
  double betop = clustertools.eTop(*b);
  double bebottom = clustertools.eBottom(*b);
  double beleft = clustertools.eLeft(*b);
  double beright = clustertools.eRight(*b);

  double be2x5max = clustertools.e2x5Max(*b);
  double be2x5top = clustertools.e2x5Top(*b);
  double be2x5bottom = clustertools.e2x5Bottom(*b);
  double be2x5left = clustertools.e2x5Left(*b);
  double be2x5right = clustertools.e2x5Right(*b);

  fVals[11] = b->eta() - s->eta();
  fVals[12] = reco::deltaPhi(b->phi(), s->phi());
  fVals[13] = b->energy() / s->rawEnergy();
  fVals[14] = clustertools.e3x3(*b) / b->energy();
  fVals[15] = clustertools.e5x5(*b) / b->energy();
  fVals[16] = sqrt(clustertools.localCovariances(*b)[0]);  //sigietaieta
  fVals[17] = sqrt(clustertools.localCovariances(*b)[2]);  //sigiphiiphi
  fVals[18] = clustertools.localCovariances(*b)[1];        //sigietaiphi
  fVals[19] = bemax / b->energy();                         //crystal energy ratio gap variables
  fVals[20] = be2nd / b->energy();
  fVals[21] = betop / b->energy();
  fVals[22] = bebottom / b->energy();
  fVals[23] = beleft / b->energy();
  fVals[24] = beright / b->energy();
  fVals[25] = be2x5max / b->energy();  //crystal energy ratio gap variables
  fVals[26] = be2x5top / b->energy();
  fVals[27] = be2x5bottom / b->energy();
  fVals[28] = be2x5left / b->energy();
  fVals[29] = be2x5right / b->energy();

  if (isbarrel) {
    //local coordinates and crystal indices (barrel only)

    //seed cluster
    float betacry, bphicry, bthetatilt, bphitilt;
    int bieta, biphi;
    egammaTools::localEcalClusterCoordsEB(*b, caloGeometry, betacry, bphicry, bieta, biphi, bthetatilt, bphitilt);

    fVals[30] = bieta;      //crystal ieta
    fVals[31] = biphi;      //crystal iphi
    fVals[32] = bieta % 5;  //submodule boundary eta symmetry
    fVals[33] = biphi % 2;  //submodule boundary phi symmetry
    fVals[34] = (TMath::Abs(bieta) <= 25) * (bieta % 25) +
                (TMath::Abs(bieta) > 25) *
                    ((bieta - 25 * TMath::Abs(bieta) / bieta) % 20);  //module boundary eta approximate symmetry
    fVals[35] = biphi % 20;                                           //module boundary phi symmetry
    fVals[36] = betacry;  //local coordinates with respect to closest crystal center at nominal shower depth
    fVals[37] = bphicry;

  } else {
    //preshower energy ratio (endcap only)
    fVals[30] = s->preshowerEnergy() / s->rawEnergy();
  }

  //   if (isbarrel) {
  //     for (int i=0; i<38; ++i) printf("%i: %5f\n",i,fVals[i]);
  //   }
  //   else for (int i=0; i<31; ++i) printf("%i: %5f\n",i,fVals[i]);

  Double_t den;
  const GBRForest *reader;
  const GBRForest *readervar;
  if (isbarrel) {
    den = s->rawEnergy();
    reader = fReadereb;
    readervar = fReaderebvariance;
  } else {
    den = s->rawEnergy() + s->preshowerEnergy();
    reader = fReaderee;
    readervar = fReadereevariance;
  }

  Double_t ecor = reader->GetResponse(fVals.data()) * den;

  //apply shower shape rescaling - for Monte Carlo only, and only for calculation of energy uncertainty
  if (applyRescale) {
    if (isbarrel) {
      fVals[3] = 1.0045 * p.r9() + 0.001;                   //r9
      fVals[5] = 1.04302 * s->etaWidth() - 0.000618;        //etawidth
      fVals[6] = 1.00002 * s->phiWidth() - 0.000371;        //phiwidth
      fVals[14] = fVals[3] * s->rawEnergy() / b->energy();  //compute consistent e3x3/eseed after r9 rescaling
      if (fVals[15] <= 1.0)  // rescale e5x5/eseed only if value is <=1.0, don't allow scaled values to exceed 1.0
        fVals[15] = TMath::Min(1.0, 1.0022 * p.e5x5() / b->energy());

      fVals[4] =
          fVals[15] * b->energy() / s->rawEnergy();  // compute consistent e5x5()/rawEnergy() after e5x5/eseed resacling

      fVals[16] = 0.891832 * sqrt(clustertools.localCovariances(*b)[0]) + 0.0009133;  //sigietaieta
      fVals[17] = 0.993 * sqrt(clustertools.localCovariances(*b)[2]);                 //sigiphiiphi

      fVals[19] = 1.012 * bemax / b->energy();  //crystal energy ratio gap variables
      fVals[20] = 1.0 * be2nd / b->energy();
      fVals[21] = 0.94 * betop / b->energy();
      fVals[22] = 0.94 * bebottom / b->energy();
      fVals[23] = 0.94 * beleft / b->energy();
      fVals[24] = 0.94 * beright / b->energy();
      fVals[25] = 1.006 * be2x5max / b->energy();  //crystal energy ratio gap variables
      fVals[26] = 1.09 * be2x5top / b->energy();
      fVals[27] = 1.09 * be2x5bottom / b->energy();
      fVals[28] = 1.09 * be2x5left / b->energy();
      fVals[29] = 1.09 * be2x5right / b->energy();

    } else {
      fVals[3] = 1.0086 * p.r9() - 0.0007;                             //r9
      fVals[4] = TMath::Min(1.0, 1.0022 * p.e5x5() / s->rawEnergy());  //e5x5/rawenergy
      fVals[5] = 0.903254 * s->etaWidth() + 0.001346;                  //etawidth
      fVals[6] = 0.99992 * s->phiWidth() + 4.8e-07;                    //phiwidth
      fVals[13] =
          TMath::Min(1.0, 1.0022 * b->energy() / s->rawEnergy());  //eseed/rawenergy (practically equivalent to e5x5)

      fVals[14] = fVals[3] * s->rawEnergy() / b->energy();  //compute consistent e3x3/eseed after r9 rescaling

      fVals[16] = 0.9947 * sqrt(clustertools.localCovariances(*b)[0]) + 0.00003;  //sigietaieta

      fVals[19] = 1.005 * bemax / b->energy();  //crystal energy ratio gap variables
      fVals[20] = 1.02 * be2nd / b->energy();
      fVals[21] = 0.96 * betop / b->energy();
      fVals[22] = 0.96 * bebottom / b->energy();
      fVals[23] = 0.96 * beleft / b->energy();
      fVals[24] = 0.96 * beright / b->energy();
      fVals[25] = 1.0075 * be2x5max / b->energy();  //crystal energy ratio gap variables
      fVals[26] = 1.13 * be2x5top / b->energy();
      fVals[27] = 1.13 * be2x5bottom / b->energy();
      fVals[28] = 1.13 * be2x5left / b->energy();
      fVals[29] = 1.13 * be2x5right / b->energy();
    }
  }

  Double_t ecorerr = readervar->GetResponse(fVals.data()) * den;

  //printf("ecor = %5f, ecorerr = %5f\n",ecor,ecorerr);

  return {ecor, ecorerr};
}
