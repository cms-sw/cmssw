#include "RecoEgamma/ElectronIdentification/interface/CutBasedElectronID.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "RecoEgamma/EgammaTools/interface/ConversionFinder.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/Scalers/interface/DcsStatus.h"

#include <algorithm>

void CutBasedElectronID::setup(const edm::ParameterSet& conf) {

  // Get all the parameters
  baseSetup(conf);

  type_ = conf.getParameter<std::string>("electronIDType");
  quality_ = conf.getParameter<std::string>("electronQuality");
  version_ = conf.getParameter<std::string>("electronVersion");
  verticesCollection_ = conf.getParameter<edm::InputTag>("verticesCollection");

  dataMagneticFieldSetUp_ = conf.getParameter<Bool_t>("dataMagneticFieldSetUp");
  if (dataMagneticFieldSetUp_) {
    dcsTag_ = conf.getParameter<edm::InputTag>("dcsTag");
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

  } else if (version_ == "V03" or version_ == "V04" or version_ == "V05" or version_ == "") {
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

  double eta = electron->p4().Eta();
  double eOverP = electron->eSuperClusterOverP();
  //double eSeed = electron->superCluster()->seed()->energy();
  //double pin  = electron->trackMomentumAtVtx().R();
  double eSeedOverPin = electron->eSeedClusterOverP();
  double fBrem = electron->fbrem();
  double hOverE = electron->hadronicOverEm();
  //EcalClusterLazyTools lazyTools = getClusterShape(e,es);
  //std::vector<float> vLocCov = lazyTools.localCovariances(*(electron->superCluster()->seed())) ;
  double sigmaee = electron->sigmaIetaIeta(); //sqrt(vLocCov[0]);
  double deltaPhiIn = electron->deltaPhiSuperClusterTrackAtVtx();
  double deltaEtaIn = electron->deltaEtaSuperClusterTrackAtVtx();

  double ip = 0;
  int mishits = electron->gsfTrack()->trackerExpectedHitsInner().numberOfHits();
  double tkIso = electron->dr03TkSumPt();
  double ecalIso = electron->dr04EcalRecHitSumEt();
  double hcalIso = electron->dr04HcalTowerSumEt();

  // calculate the conversion track partner related criteria
  const math::XYZPoint tpoint = electron->gsfTrack()->referencePoint();
  // calculate the magnetic field for that point
  Double_t bfield = 0;
  if (dataMagneticFieldSetUp_) {
    edm::Handle<DcsStatusCollection> dcsHandle;
    e.getByLabel(dcsTag_, dcsHandle);
    // scale factor = 3.801/18166.0 which are
    // average values taken over a stable two
    // week period
    Double_t currentToBFieldScaleFactor = 2.09237036221512717e-04;
    Double_t current = (*dcsHandle)[0].magnetCurrent();
    bfield = current*currentToBFieldScaleFactor;
  } else {
    edm::ESHandle<MagneticField> magneticField;
    es.get<IdealMagneticFieldRecord>().get(magneticField);
    const  MagneticField *mField = magneticField.product();
    bfield = mField->inTesla(GlobalPoint(0.,0.,0.)).z();
  }

  edm::Handle<reco::TrackCollection> ctfTracks;
  e.getByLabel("generalTracks", ctfTracks);
  ConversionFinder convFinder;

  if (version_ == "V00") {
    //std::vector<float> vCov = lazyTools.covariances(*(electron->superCluster()->seed())) ;
    sigmaee = electron->sigmaEtaEta();//sqrt(vCov[0]);
    if (electron->isEE())
      sigmaee = sigmaee - 0.02*(fabs(eta) - 2.3);   //correct sigmaetaeta dependence on eta in endcap
  }

  if (version_ != "V01" or version_ != "V00") {
    edm::Handle<reco::VertexCollection> vtxH;
    e.getByLabel(verticesCollection_, vtxH);
    if (vtxH->size() != 0) {
      reco::VertexRef vtx(vtxH, 0);
      ip = fabs(electron->gsfTrack()->dxy(math::XYZPoint(vtx->x(),vtx->y(),vtx->z())));
    } else
      ip = fabs(electron->gsfTrack()->dxy());

    if (electron->isEB()) {
      //std::vector<float> vCov = lazyTools.scLocalCovariances(*(electron->superCluster()));
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
      //std::cout << "hoe" << hOverE << std::endl;
      std::vector<double> cuthoe = cuts_.getParameter<std::vector<double> >("cuthoe");
      //std::cout << "see" << sigmaee << std::endl;
      std::vector<double> cutsee = cuts_.getParameter<std::vector<double> >("cutsee");
      //std::cout << "dphiin" << fabs(deltaPhiIn) << std::endl;
      std::vector<double> cutdphi = cuts_.getParameter<std::vector<double> >("cutdphiin");
      //std::cout << "detain" << fabs(deltaEtaIn) << std::endl;
      std::vector<double> cutdeta = cuts_.getParameter<std::vector<double> >("cutdetain");
      //std::cout << "eseedopin " << eSeedOverPin << std::endl;
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

  if (version_ == "V03" or version_ == "V04" or version_ == "V05" or version_ == "") {
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
      //std::cout << "hoe" << hOverE << std::endl;
      std::vector<double> cuthoe = cuts_.getParameter<std::vector<double> >("cuthoe");
      //std::cout << "see" << sigmaee << std::endl;
      std::vector<double> cutsee = cuts_.getParameter<std::vector<double> >("cutsee");
      //std::cout << "dphiin" << fabs(deltaPhiIn) << std::endl;
      std::vector<double> cutdphi = cuts_.getParameter<std::vector<double> >("cutdphiin");
      //std::cout << "detain" << fabs(deltaEtaIn) << std::endl;
      std::vector<double> cutdeta = cuts_.getParameter<std::vector<double> >("cutdetain");
      //std::cout << "eseedopin " << eSeedOverPin << std::endl;
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

    //std::cout << "ip" << ip << std::endl;
    std::vector<double> cutip = cuts_.getParameter<std::vector<double> >("cutip_gsf");
    if (ip < cutip[cat+bin*9])
      result += 8;

    std::vector<double> cutmishits = cuts_.getParameter<std::vector<double> >("cutfmishits");
    std::vector<double> cutdcotdist = cuts_.getParameter<std::vector<double> >("cutdcotdist");
    ConversionInfo convInfo = convFinder.getConversionInfo(*electron, ctfTracks, bfield);

    float dist = (convInfo.dist() == -9999.? 9999:convInfo.dist());
    float dcot = (convInfo.dcot() == -9999.? 9999:convInfo.dcot());

    float dcotdistcomb = ((0.04 - std::max(dist, dcot)) > 0?(0.04 - std::max(dist, dcot)):0);

    if ((mishits < cutmishits[cat+bin*9]) and
        (dcotdistcomb < cutdcotdist[cat+bin*9]))
      result += 4;

    return result;
  }

  return -1.;
}

double CutBasedElectronID::robustSelection(const reco::GsfElectron* electron ,
                                           const edm::Event& e ,
                                           const edm::EventSetup& es) {

  double scTheta = (2*atan(exp(-electron->superCluster()->eta())));
  double scEt = electron->superCluster()->energy()*sin(scTheta);
  double eta = electron->p4().Eta();
  double eOverP = electron->eSuperClusterOverP();
  double hOverE = electron->hadronicOverEm();
  //EcalClusterLazyTools lazyTools = getClusterShape(e,es);
  //std::vector<float> vLocCov = lazyTools.localCovariances(*(electron->superCluster()->seed())) ;
  double sigmaee = electron->sigmaIetaIeta();//sqrt(vLocCov[0]);
  double e25Max = electron->e2x5Max();//lazyTools.e2x5Max(*(electron->superCluster()->seed()))  ;
  double e15 = electron->e1x5();//lazyTools.e1x5(*(electron->superCluster()->seed()))  ;
  double e55 = electron->e5x5();//lazyTools.e5x5(*(electron->superCluster()->seed())) ;
  double e25Maxoe55 = e25Max/e55 ;
  double e15oe55 = e15/e55 ;
  double deltaPhiIn = electron->deltaPhiSuperClusterTrackAtVtx();
  double deltaEtaIn = electron->deltaEtaSuperClusterTrackAtVtx();

  double ip = 0;
  int mishits = electron->gsfTrack()->trackerExpectedHitsInner().numberOfHits();
  double tkIso = electron->dr03TkSumPt();
  double ecalIso = electron->dr04EcalRecHitSumEt();
  double ecalIsoPed = (electron->isEB())?std::max(0.,ecalIso-1.):ecalIso;
  double hcalIso = electron->dr04HcalTowerSumEt();
  double hcalIso1 = electron->dr04HcalDepth1TowerSumEt();
  double hcalIso2 = electron->dr04HcalDepth2TowerSumEt();

  // calculate the conversion track partner related criteria
  // calculate the reference point of the track
  const math::XYZPoint tpoint = electron->gsfTrack()->referencePoint();
  // calculate the magnetic field for that point
  Double_t bfield = 0;
  if (dataMagneticFieldSetUp_) {
    edm::Handle<DcsStatusCollection> dcsHandle;
    e.getByLabel(dcsTag_, dcsHandle);
    // scale factor = 3.801/18166.0 which are
    // average values taken over a stable two
    // week period
    Double_t currentToBFieldScaleFactor = 2.09237036221512717e-04;
    Double_t current = (*dcsHandle)[0].magnetCurrent();
    bfield = current*currentToBFieldScaleFactor;
  } else {
    edm::ESHandle<MagneticField> magneticField;
    es.get<IdealMagneticFieldRecord>().get(magneticField);
    const  MagneticField *mField = magneticField.product();
    bfield = mField->inTesla(GlobalPoint(0.,0.,0.)).z();
  }

  edm::Handle<reco::TrackCollection> ctfTracks;
  e.getByLabel("generalTracks", ctfTracks);
  ConversionFinder convFinder;

  if (version_ == "V00") {
    //std::vector<float> vCov = lazyTools.covariances(*(electron->superCluster()->seed())) ;
    sigmaee = electron->sigmaEtaEta();//sqrt(vCov[0]);
     if (electron->isEE())
       sigmaee = sigmaee - 0.02*(fabs(eta) - 2.3);   //correct sigmaetaeta dependence on eta in endcap
  }

  if (version_ == "V03" or version_ == "V04") {
    edm::Handle<reco::BeamSpot> pBeamSpot;
    // uses the same name for the vertex collection to avoid adding more new names
    e.getByLabel(verticesCollection_, pBeamSpot);
    if (pBeamSpot.isValid()) {
      const reco::BeamSpot *bspot = pBeamSpot.product();
      const math::XYZPoint bspotPosition = bspot->position();
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
    e.getByLabel(verticesCollection_, vtxH);
    if (vtxH->size() != 0) {
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

      //if (electron->isEB() && (version_ == "V03" || version_ == "V04" || version_ == ""))
      //  ecalIso = std::max(0., ecalIso - 1.);

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
    if (cut[23] >0 && not (electron->gsfTrack()->hitPattern().hasValidHitInFirstPixelBarrel()))
      return result;
    // cut[24]: Dist cut[25]: dcot
    ConversionInfo convInfo = convFinder.getConversionInfo(*electron, ctfTracks, bfield);

    if (convFinder.isFromConversion(convInfo,cut[24],cut[25]))
      return result ;

    result += 4 ;

    return result ;
   }

  return -1. ;
 }
