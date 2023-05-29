#ifndef Alignment_OfflineValidation_EopElecVariables_h
#define Alignment_OfflineValidation_EopElecVariables_h

// For ROOT types with '_t':
#include <Rtypes.h>

// container to hold data to be written into TTree
struct EopElecVariables {
  /// constructor initialises to empty values
  EopElecVariables() { clear(); }
  ~EopElecVariables() = default;
  /// set to empty values
  void clear() {
    /// doubles
    outerRadius = chi2 = normalizedChi2 = p = pt = ptError = theta = eta = phi = SC_energy = HcalEnergyIn01 =
        HcalEnergyIn02 = HcalEnergyIn03 = HcalEnergyIn04 = HcalEnergyIn05 = SC_etaWidth = SC_phiWidth = fbrem = SC_eta =
            SC_phi = pIn = pOut = etaIn = phiIn = etaOut = phiOut = px = py = pz = dRto1stSC = dRto2ndSC = 0.;
    /// integers
    charge = nHits = nLostHits = SC_nBasicClus = SC_algoID = RunNumber = EvtNumber = 0;
    /// booleans
    innerOk = isEcalDriven = isTrackerDriven = SC_isBarrel = SC_isEndcap = false;

    MaxPtIn01 = 0.;
    SumPtIn01 = 0.;
    NoTrackIn0015 = true;
    MaxPtIn02 = 0.;
    SumPtIn02 = 0.;
    NoTrackIn0020 = true;
    MaxPtIn03 = 0.;
    SumPtIn03 = 0.;
    NoTrackIn0025 = true;
    MaxPtIn04 = 0.;
    SumPtIn04 = 0.;
    NoTrackIn0030 = true;
    MaxPtIn05 = 0.;
    SumPtIn05 = 0.;
    NoTrackIn0035 = true;
    NoTrackIn0040 = true;

    px_rejected_track = 0.;
    py_rejected_track = 0.;
    pz_rejected_track = 0.;
    p_rejected_track = 0.;
  }

  Int_t charge;
  Int_t nHits;
  Int_t nLostHits;
  Bool_t innerOk;
  Double_t outerRadius;
  Double_t chi2;
  Double_t normalizedChi2;
  Double_t px_rejected_track;
  Double_t py_rejected_track;
  Double_t pz_rejected_track;
  Double_t p_rejected_track;
  Double_t px;
  Double_t py;
  Double_t pz;
  Double_t p;
  Double_t pIn;
  Double_t etaIn;
  Double_t phiIn;
  Double_t pOut;
  Double_t etaOut;
  Double_t phiOut;
  Double_t pt;
  Double_t ptError;
  Double_t theta;
  Double_t eta;
  Double_t phi;
  Double_t fbrem;
  Double_t MaxPtIn01;
  Double_t SumPtIn01;
  Bool_t NoTrackIn0015;
  Double_t MaxPtIn02;
  Double_t SumPtIn02;
  Bool_t NoTrackIn0020;
  Double_t MaxPtIn03;
  Double_t SumPtIn03;
  Bool_t NoTrackIn0025;
  Double_t MaxPtIn04;
  Double_t SumPtIn04;
  Bool_t NoTrackIn0030;
  Double_t MaxPtIn05;
  Double_t SumPtIn05;
  Bool_t NoTrackIn0035;
  Double_t NoTrackIn0040;
  Int_t SC_algoID;
  Double_t SC_energy;
  Int_t SC_nBasicClus;
  Double_t SC_etaWidth;
  Double_t SC_phiWidth;
  Double_t SC_eta;
  Double_t SC_phi;
  Bool_t SC_isBarrel;
  Bool_t SC_isEndcap;
  Double_t dRto1stSC;
  Double_t dRto2ndSC;
  Double_t HcalEnergyIn01;
  Double_t HcalEnergyIn02;
  Double_t HcalEnergyIn03;
  Double_t HcalEnergyIn04;
  Double_t HcalEnergyIn05;
  Bool_t isEcalDriven;
  Bool_t isTrackerDriven;
  Int_t RunNumber;
  Int_t EvtNumber;
};

#endif
