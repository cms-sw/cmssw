#include "RecoEgamma/ElectronIdentification/interface/CutBasedElectronID.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
//#include "RecoEgamma/EgammaTools/interface/ConversionFinder.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include <algorithm>

CutBasedElectronID::CutBasedElectronID(const edm::ParameterSet& conf,edm::ConsumesCollector & iC)
{
  verticesCollection_ = iC.consumes<std::vector<reco::Vertex> >(conf.getParameter<edm::InputTag>("verticesCollection"));

}

void CutBasedElectronID::setup(const edm::ParameterSet& conf) {

  // Get all the parameters
  //baseSetup(conf);

  type_ = conf.getParameter<std::string>("electronIDType");
  quality_ = conf.getParameter<std::string>("electronQuality");
  version_ = conf.getParameter<std::string>("electronVersion");
  //verticesCollection_ = conf.getParameter<edm::InputTag>("verticesCollection");
  
  if (type_ == "classbased" and (version_ == "V06")) {
    newCategories_ = conf.getParameter<bool>("additionalCategories");
  }

  if (type_ == "classbased" and (version_ == "V03" or version_ == "V04" or version_ == "V05" or version_ == "")) {
    wantBinning_ = conf.getParameter<bool>("etBinning");
    newCategories_ = conf.getParameter<bool>("additionalCategories");
  }

  if (type_ == "robust" || type_ == "classbased") {
    std::string stringCut = type_+quality_+"EleIDCuts"+version_;
    cuts_ = conf.getParameter<edm::ParameterSet>(stringCut);
  }
  else {
    throw cms::Exception("Configuration")
      << "Invalid electronType parameter in CutBasedElectronID: must be robust or classbased\n";
  }
}

double CutBasedElectronID::result(const reco::GsfElectron* electron ,
                                  const edm::Event& e ,
                                  const edm::EventSetup& es) {

  if (type_ == "classbased")
    return cicSelection(electron, e, es);
  else if (type_ == "robust")
    return robustSelection(electron, e, es);

  return 0;

}

int CutBasedElectronID::classify(const reco::GsfElectron* electron) {

  double eta = fabs(electron->superCluster()->eta());
  double eOverP = electron->eSuperClusterOverP();
  double fBrem = electron->fbrem();

  int cat = -1;
  if (version_ == "V00" || version_ == "V01") {
    if((electron->isEB() && fBrem<0.06) || (electron->isEE() && fBrem<0.1))
      cat=1;
    else if (eOverP < 1.2 && eOverP > 0.8)
      cat=0;
    else
      cat=2;

    return cat;

  } else if (version_ == "V02") {
    if (electron->isEB()) {       // BARREL
      if(fBrem < 0.12)
        cat=1;
      else if (eOverP < 1.2 && eOverP > 0.9)
        cat=0;
      else
        cat=2;
    } else {                     // ENDCAP
      if(fBrem < 0.2)
        cat=1;
      else if (eOverP < 1.22 && eOverP > 0.82)
        cat=0;
      else
        cat=2;
    }

    return cat;

  } else {
    if (electron->isEB()) {
      if ((fBrem >= 0.12) and (eOverP > 0.9) and (eOverP < 1.2))
        cat = 0;
      else if (((eta >  .445   and eta <  .45  ) or
                (eta >  .79    and eta <  .81  ) or
                (eta > 1.137   and eta < 1.157 ) or
                (eta > 1.47285 and eta < 1.4744)) and newCategories_)
        cat = 6;
      else if (electron->trackerDrivenSeed() and !electron->ecalDrivenSeed() and newCategories_)
        cat = 8;
      else if (fBrem < 0.12)
        cat = 1;
      else
        cat = 2;
    } else {
      if ((fBrem >= 0.2) and (eOverP > 0.82) and (eOverP < 1.22))
        cat = 3;
      else if (eta > 1.5 and eta <  1.58 and newCategories_)
        cat = 7;
      else if (electron->trackerDrivenSeed() and !electron->ecalDrivenSeed() and newCategories_)
        cat = 8;
      else if (fBrem < 0.2)
        cat = 4;
      else
        cat = 5;
    }

    return cat;
  }

  return -1;
}

double CutBasedElectronID::cicSelection(const reco::GsfElectron* electron,
                                        const edm::Event& e,
                                        const edm::EventSetup& es) {

  double scTheta = (2*atan(exp(-electron->superCluster()->eta())));
  double scEt = electron->superCluster()->energy()*sin(scTheta);

  double eta = fabs(electron->superCluster()->eta());

  double eOverP = electron->eSuperClusterOverP();
  double eSeedOverPin = electron->eSeedClusterOverP();
  double fBrem = electron->fbrem();
  double hOverE = electron->hadronicOverEm();
  double sigmaee = electron->sigmaIetaIeta(); //sqrt(vLocCov[0]);
  double deltaPhiIn = electron->deltaPhiSuperClusterTrackAtVtx();
  double deltaEtaIn = electron->deltaEtaSuperClusterTrackAtVtx();

  double ip = 0;
  int mishits = electron->gsfTrack()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
  double tkIso = electron->dr03TkSumPt();
  double ecalIso = electron->dr04EcalRecHitSumEt();
  double hcalIso = electron->dr04HcalTowerSumEt();

  if (version_ == "V00") {
    sigmaee = electron->sigmaEtaEta();//sqrt(vCov[0]);
    if (electron->isEE())
      sigmaee = sigmaee - 0.02*(fabs(eta) - 2.3);   //correct sigmaetaeta dependence on eta in endcap
  }

  if (version_ != "V01" or version_ != "V00") {
    edm::Handle<reco::VertexCollection> vtxH;
    e.getByToken(verticesCollection_, vtxH);
    if (!vtxH->empty()) {
      reco::VertexRef vtx(vtxH, 0);
      ip = fabs(electron->gsfTrack()->dxy(math::XYZPoint(vtx->x(),vtx->y(),vtx->z())));
    } else
      ip = fabs(electron->gsfTrack()->dxy());

    if (electron->isEB()) {
      sigmaee = electron->sigmaIetaIeta(); //sqrt(vCov[0]);
    }
  }

  std::vector<double> cut;

  int cat = classify(electron);
  int eb;

  if (electron->isEB())
    eb = 0;
  else
    eb = 1;

  // LOOSE and TIGHT Selections
  if (type_ == "classbased" && (version_ == "V01" || version_ == "V00")) {

    if ((eOverP < 0.8) && (fBrem < 0.2))
      return 0.;

    cut = cuts_.getParameter<std::vector<double> >("hOverE");
    if (hOverE > cut[cat+4*eb])
      return 0.;

    cut = cuts_.getParameter<std::vector<double> >("sigmaEtaEta");
    if (sigmaee > cut[cat+4*eb])
      return 0.;

    cut = cuts_.getParameter<std::vector<double> >("deltaPhiIn");
    if (eOverP < 1.5) {
      if (fabs(deltaPhiIn) > cut[cat+4*eb])
        return 0.;
    } else {
      if (fabs(deltaPhiIn) > cut[3+4*eb])
        return 0.;
    }

    cut = cuts_.getParameter<std::vector<double> >("deltaEtaIn");
    if (fabs(deltaEtaIn) > cut[cat+4*eb])
      return 0.;

    cut = cuts_.getParameter<std::vector<double> >("eSeedOverPin");
    if (eSeedOverPin < cut[cat+4*eb])
      return 0.;

    if (quality_ == "tight")
      if (eOverP < 0.9*(1-fBrem))
        return 0.;

    return 1.;
  }

  if (type_ == "classbased" and version_ == "V02") {
    double result = 0.;

    int bin = 0;

    if (scEt < 20.)
      bin = 2;
    else if (scEt > 30.)
      bin = 0;
    else
      bin = 1;

    if (fBrem > 0)
      eSeedOverPin = eSeedOverPin + fBrem;

    if (bin != 2) {
      tkIso = tkIso*pow(40./scEt, 2);
      ecalIso = ecalIso*pow(40./scEt, 2);
      hcalIso = hcalIso*pow(40./scEt, 2);
    }

    std::vector<double> cutTk = cuts_.getParameter<std::vector<double> >("cutisotk");
    std::vector<double> cutEcal = cuts_.getParameter<std::vector<double> >("cutisoecal");
    std::vector<double> cutHcal = cuts_.getParameter<std::vector<double> >("cutisohcal");
    if ((tkIso > cutTk[cat+3*eb+bin*6]) ||
        (ecalIso > cutEcal[cat+3*eb+bin*6]) ||
        (hcalIso > cutHcal[cat+3*eb+bin*6]))
      result = 0.;
    else
      result = 2.;

    if (fBrem > -2) {
      std::vector<double> cuthoe = cuts_.getParameter<std::vector<double> >("cuthoe");
      std::vector<double> cutsee = cuts_.getParameter<std::vector<double> >("cutsee");
      std::vector<double> cutdphi = cuts_.getParameter<std::vector<double> >("cutdphiin");
      std::vector<double> cutdeta = cuts_.getParameter<std::vector<double> >("cutdetain");
      std::vector<double> cuteopin = cuts_.getParameter<std::vector<double> >("cuteseedopcor");
      std::vector<double> cutet = cuts_.getParameter<std::vector<double> >("cutet");
      std::vector<double> cutip = cuts_.getParameter<std::vector<double> >("cutip");
      std::vector<double> cutmishits = cuts_.getParameter<std::vector<double> >("cutmishits");
      if ((hOverE < cuthoe[cat+3*eb+bin*6]) and
          (sigmaee < cutsee[cat+3*eb+bin*6]) and
          (fabs(deltaPhiIn) < cutdphi[cat+3*eb+bin*6]) and
          (fabs(deltaEtaIn) < cutdeta[cat+3*eb+bin*6]) and
          (eSeedOverPin > cuteopin[cat+3*eb+bin*6]) and
          (ip < cutip[cat+3*eb+bin*6]) and
          (mishits < cutmishits[cat+3*eb+bin*6]))
        result = result + 1.;
    }
    return result;
  }

  if (version_ == "V03" or version_ == "V04" or version_ == "V05") {
    double result = 0.;

    int bin = 0;

    if (wantBinning_) {
      if (scEt < 20.)
        bin = 2;
      else if (scEt > 30.)
        bin = 0;
      else
        bin = 1;
    }

    if (fBrem > 0)
      eSeedOverPin = eSeedOverPin + fBrem;

    float iso_sum = tkIso + ecalIso + hcalIso;
    float iso_sum_corrected = iso_sum*pow(40./scEt, 2);

    std::vector<double> cutIsoSum = cuts_.getParameter<std::vector<double> >("cutiso_sum");
    std::vector<double> cutIsoSumCorr = cuts_.getParameter<std::vector<double> >("cutiso_sumoet");
    if ((iso_sum < cutIsoSum[cat+bin*9]) and
        (iso_sum_corrected < cutIsoSumCorr[cat+bin*9]))
      result += 2.;

    if (fBrem > -2) {
      std::vector<double> cuthoe = cuts_.getParameter<std::vector<double> >("cuthoe");
      std::vector<double> cutsee = cuts_.getParameter<std::vector<double> >("cutsee");
      std::vector<double> cutdphi = cuts_.getParameter<std::vector<double> >("cutdphiin");
      std::vector<double> cutdeta = cuts_.getParameter<std::vector<double> >("cutdetain");
      std::vector<double> cuteopin = cuts_.getParameter<std::vector<double> >("cuteseedopcor");
      std::vector<double> cutet = cuts_.getParameter<std::vector<double> >("cutet");

      if ((hOverE < cuthoe[cat+bin*9]) and
          (sigmaee < cutsee[cat+bin*9]) and
          (fabs(deltaPhiIn) < cutdphi[cat+bin*9]) and
          (fabs(deltaEtaIn) < cutdeta[cat+bin*9]) and
          (eSeedOverPin > cuteopin[cat+bin*9]) and
          (scEt > cutet[cat+bin*9]))
        result += 1.;
    }

    std::vector<double> cutip = cuts_.getParameter<std::vector<double> >("cutip_gsf");
    if (ip < cutip[cat+bin*9])
      result += 8;
    
    std::vector<double> cutmishits = cuts_.getParameter<std::vector<double> >("cutfmishits");
    std::vector<double> cutdcotdist = cuts_.getParameter<std::vector<double> >("cutdcotdist");
    
    float dist = (electron->convDist() == -9999.? 9999:electron->convDist());
    float dcot = (electron->convDcot() == -9999.? 9999:electron->convDcot());

    float dcotdistcomb = ((0.04 - std::max(fabs(dist), fabs(dcot))) > 0?(0.04 - std::max(fabs(dist), fabs(dcot))):0);

    if ((mishits < cutmishits[cat+bin*9]) and
        (dcotdistcomb < cutdcotdist[cat+bin*9]))
      result += 4;

    return result;
  }

  if (type_ == "classbased" && (version_ == "V06" || version_ == "")) { 
    std::vector<double> cutIsoSum      = cuts_.getParameter<std::vector<double> >("cutiso_sum");
    std::vector<double> cutIsoSumCorr  = cuts_.getParameter<std::vector<double> >("cutiso_sumoet");
    std::vector<double> cuthoe         = cuts_.getParameter<std::vector<double> >("cuthoe");
    std::vector<double> cutsee         = cuts_.getParameter<std::vector<double> >("cutsee");
    std::vector<double> cutdphi        = cuts_.getParameter<std::vector<double> >("cutdphiin");
    std::vector<double> cutdeta        = cuts_.getParameter<std::vector<double> >("cutdetain");
    std::vector<double> cuteopin       = cuts_.getParameter<std::vector<double> >("cuteseedopcor");
    std::vector<double> cutmishits     = cuts_.getParameter<std::vector<double> >("cutfmishits");
    std::vector<double> cutdcotdist    = cuts_.getParameter<std::vector<double> >("cutdcotdist");
    std::vector<double> cutip          = cuts_.getParameter<std::vector<double> >("cutip_gsf");
    std::vector<double> cutIsoSumCorrl = cuts_.getParameter<std::vector<double> >("cutiso_sumoetl");
    std::vector<double> cuthoel        = cuts_.getParameter<std::vector<double> >("cuthoel");
    std::vector<double> cutseel        = cuts_.getParameter<std::vector<double> >("cutseel");
    std::vector<double> cutdphil       = cuts_.getParameter<std::vector<double> >("cutdphiinl");
    std::vector<double> cutdetal       = cuts_.getParameter<std::vector<double> >("cutdetainl");
    std::vector<double> cutipl         = cuts_.getParameter<std::vector<double> >("cutip_gsfl");
    
    int result = 0;
    
    const int ncuts = 10;
    std::vector<bool> cut_results(ncuts, false);
    
    float iso_sum = tkIso + ecalIso + hcalIso;
    float scEta = electron->superCluster()->eta();
    if(fabs(scEta)>1.5) 
      iso_sum += (fabs(scEta)-1.5)*1.09;
    
    float iso_sumoet = iso_sum*(40./scEt);
    
    float eseedopincor = eSeedOverPin + fBrem;
    if(fBrem < 0)
      eseedopincor = eSeedOverPin;

    float dist = (electron->convDist() == -9999.? 9999:electron->convDist());
    float dcot = (electron->convDcot() == -9999.? 9999:electron->convDcot());

    float dcotdistcomb = ((0.04 - std::max(fabs(dist), fabs(dcot))) > 0?(0.04 - std::max(fabs(dist), fabs(dcot))):0);

    for (int cut=0; cut<ncuts; cut++) {
      switch (cut) {
      case 0:
        cut_results[cut] = compute_cut(fabs(deltaEtaIn), scEt, cutdetal[cat], cutdeta[cat]);
        break;
      case 1:
        cut_results[cut] = compute_cut(fabs(deltaPhiIn), scEt, cutdphil[cat], cutdphi[cat]);
        break;
      case 2:
        cut_results[cut] = (eseedopincor > cuteopin[cat]);
        break;
      case 3:
        cut_results[cut] = compute_cut(hOverE, scEt, cuthoel[cat], cuthoe[cat]);
        break;
      case 4:
        cut_results[cut] = compute_cut(sigmaee, scEt, cutseel[cat], cutsee[cat]);
        break;
      case 5:
        cut_results[cut] = compute_cut(iso_sumoet, scEt, cutIsoSumCorrl[cat], cutIsoSumCorr[cat]);
        break;
      case 6:
        cut_results[cut] = (iso_sum < cutIsoSum[cat]);
        break;
      case 7:
        cut_results[cut] = compute_cut(fabs(ip), scEt, cutipl[cat], cutip[cat]);
        break;
      case 8:
        cut_results[cut] = (mishits < cutmishits[cat]);
        break;
      case 9:
        cut_results[cut] = (dcotdistcomb < cutdcotdist[cat]);
        break;
      }
    }
    
    // ID part
    if (cut_results[0] & cut_results[1] & cut_results[2] & cut_results[3] & cut_results[4])
      result = result + 1;
    
    // ISO part
    if (cut_results[5] & cut_results[6])
      result = result + 2;
    
    // IP part
    if (cut_results[7])
      result = result + 8;
    
    // Conversion part
    if (cut_results[8] & cut_results[9])
      result = result + 4;

    return result;
  }

  return -1.;
}


bool CutBasedElectronID::compute_cut(double x, double et, double cut_min, double cut_max, bool gtn) {

  float et_min = 10;
  float et_max = 40;

  bool accept = false;
  float cut = cut_max; //  the cut at et=40 GeV

  if(et < et_max) {
    cut = cut_min + (1/et_min - 1/et)*(cut_max - cut_min)/(1/et_min - 1/et_max);
  } 
  
  if(et < et_min) {
    cut = cut_min;
  } 

  if(gtn) {   // useful for e/p cut which is gt
    accept = (x >= cut);
  } 
  else {
    accept = (x <= cut);
  }

  //std::cout << x << " " << cut_min << " " << cut << " " << cut_max << " " << et << " " << accept << std::endl;
  return accept;
}

double CutBasedElectronID::robustSelection(const reco::GsfElectron* electron ,
                                           const edm::Event& e ,
                                           const edm::EventSetup& es) {

  double scTheta = (2*atan(exp(-electron->superCluster()->eta())));
  double scEt = electron->superCluster()->energy()*sin(scTheta);
  double eta = electron->p4().Eta();
  double eOverP = electron->eSuperClusterOverP();
  double hOverE = electron->hadronicOverEm();
  double sigmaee = electron->sigmaIetaIeta();
  double e25Max = electron->e2x5Max();
  double e15 = electron->e1x5();
  double e55 = electron->e5x5();
  double e25Maxoe55 = e25Max/e55;
  double e15oe55 = e15/e55 ;
  double deltaPhiIn = electron->deltaPhiSuperClusterTrackAtVtx();
  double deltaEtaIn = electron->deltaEtaSuperClusterTrackAtVtx();

  double ip = 0;
  int mishits = electron->gsfTrack()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
  double tkIso = electron->dr03TkSumPt();
  double ecalIso = electron->dr04EcalRecHitSumEt();
  double ecalIsoPed = (electron->isEB())?std::max(0.,ecalIso-1.):ecalIso;
  double hcalIso = electron->dr04HcalTowerSumEt();
  double hcalIso1 = electron->dr04HcalDepth1TowerSumEt();
  double hcalIso2 = electron->dr04HcalDepth2TowerSumEt();

  if (version_ == "V00") {
    sigmaee = electron->sigmaEtaEta();
     if (electron->isEE())
       sigmaee = sigmaee - 0.02*(fabs(eta) - 2.3);   //correct sigmaetaeta dependence on eta in endcap
  }

  if (version_ == "V03" or version_ == "V04") {
    edm::Handle<reco::BeamSpot> pBeamSpot;
    // uses the same name for the vertex collection to avoid adding more new names
    e.getByToken(verticesCollection_, pBeamSpot);
    if (pBeamSpot.isValid()) {
      const reco::BeamSpot *bspot = pBeamSpot.product();
      const math::XYZPoint& bspotPosition = bspot->position();
      ip = fabs(electron->gsfTrack()->dxy(bspotPosition));
    } else
      ip = fabs(electron->gsfTrack()->dxy());
  }

  if (version_ == "V04" or version_ == "V05") {
    ecalIso = electron->dr03EcalRecHitSumEt();
    ecalIsoPed = (electron->isEB())?std::max(0.,ecalIso-1.):ecalIso;
    hcalIso = electron->dr03HcalTowerSumEt();
    hcalIso1 = electron->dr03HcalDepth1TowerSumEt();
    hcalIso2 = electron->dr03HcalDepth2TowerSumEt();
  }

  if (version_ == "V05") {
    edm::Handle<reco::VertexCollection> vtxH;
    e.getByToken(verticesCollection_, vtxH);
    if (!vtxH->empty()) {
      reco::VertexRef vtx(vtxH, 0);
      ip = fabs(electron->gsfTrack()->dxy(math::XYZPoint(vtx->x(),vtx->y(),vtx->z())));
    } else
      ip = fabs(electron->gsfTrack()->dxy());
  }

  // .....................................................................................
  std::vector<double> cut;
  // ROBUST Selection
  if (type_ == "robust") {

    double result = 0;

    // hoe, sigmaEtaEta, dPhiIn, dEtaIn
    if (electron->isEB())
      cut = cuts_.getParameter<std::vector<double> >("barrel");
    else
      cut = cuts_.getParameter<std::vector<double> >("endcap");
    // check isolations: if only isolation passes result = 2
    if (quality_ == "highenergy") {
      if ((tkIso > cut[6] || hcalIso2 > cut[11]) ||
          (electron->isEB() && ((ecalIso + hcalIso1) > cut[7]+cut[8]*scEt)) ||
          (electron->isEE() && (scEt >= 50.) && ((ecalIso + hcalIso1) > cut[7]+cut[8]*(scEt-50))) ||
          (electron->isEE() && (scEt < 50.) && ((ecalIso + hcalIso1) > cut[9]+cut[10]*(scEt-50))))
        result = 0;
      else
        result = 2;
    } else {
      if ((tkIso > cut[6]) || (ecalIso > cut[7]) || (hcalIso > cut[8]) || (hcalIso1 > cut[9]) || (hcalIso2 > cut[10]) ||
          (tkIso/electron->p4().Pt() > cut[11]) || (ecalIso/electron->p4().Pt() > cut[12]) || (hcalIso/electron->p4().Pt() > cut[13]) ||
          ((tkIso+ecalIso+hcalIso)>cut[14]) || (((tkIso+ecalIso+hcalIso)/ electron->p4().Pt()) > cut[15]) ||
          ((tkIso+ecalIsoPed+hcalIso)>cut[16]) || (((tkIso+ecalIsoPed+hcalIso)/ electron->p4().Pt()) > cut[17])  )
        result = 0.;
      else
        result = 2.;
    }

    if ((hOverE < cut[0]) && (sigmaee < cut[1]) && (fabs(deltaPhiIn) < cut[2]) &&
        (fabs(deltaEtaIn) < cut[3]) && (e25Maxoe55 > cut[4] && e15oe55 > cut[5]) &&
        (sigmaee >= cut[18]) && (eOverP > cut[19] &&  eOverP < cut[20]) )
     { result = result + 1 ; }

    if (ip > cut[21])
      return result;
    if (mishits > cut[22]) // expected missing hits
      return result;
    // positive cut[23] means to demand a valid hit in 1st layer PXB
    if (cut[23] > 0 && !electron->gsfTrack()->hitPattern().hasValidHitInPixelLayer(PixelSubdetector::SubDetector::PixelBarrel, 1))
      return result;

    // cut[24]: Dist cut[25]: dcot
    float dist = fabs(electron->convDist());
    float dcot = fabs(electron->convDcot());
    bool isConversion = (cut[24]>99. || cut[25]>99.)?false:(dist < cut[24] && dcot < cut[25]);
    if (isConversion)
      return result ;
    
    result += 4 ;

    return result ;
   }

  return -1. ;
 }
