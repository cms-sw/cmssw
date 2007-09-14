{
TFile signalFile("../test/ZEE-HLTEgamma.root");
TTree *sigEvents = (TTree*)signalFile.Get("Events");
TFile background5080File("../test/QCD50-80-HLTEgamma.root");
TTree *bkgEvents = (TTree*)background5080File.Get("Events");
Long64_t *sSigEt = new Long64_t[120];
Long64_t *rsSigEt = new Long64_t[120];
Long64_t *dSigEt = new Long64_t[120];
Long64_t *rdSigEt = new Long64_t[120];
Long64_t *sBkgEt = new Long64_t[120];
Long64_t *rsBkgEt = new Long64_t[120];
Long64_t *dBkgEt = new Long64_t[120];
Long64_t *rdBkgEt = new Long64_t[120];

Long64_t *sSigEtEff = new Long64_t[120];
Long64_t *rsSigEtEff = new Long64_t[120];
Long64_t *dSigEtEff = new Long64_t[120];
Long64_t *rdSigEtEff = new Long64_t[120];
Long64_t *sBkgEtRate = new Long64_t[120];
Long64_t *rsBkgEtRate = new Long64_t[120];
Long64_t *dBkgEtRate = new Long64_t[120];
Long64_t *rdBkgEtRate = new Long64_t[120];

Double_t sMinSigEtEff = 0;
Double_t rsMinSigEtEff = 0;
Double_t dMinSigEtEff = 0;
Double_t rdMinSigEtEff = 0;
Double_t sMinBkgEtRate = 0;
Double_t rsMinBkgEtRate = 0;
Double_t dMinBkgEtRate = 0;
Double_t rdMinBkgEtRate = 0;

Double_t sMaxSigEtEff = 0;
Double_t rsMaxSigEtEff = 0;
Double_t dMaxSigEtEff = 0;
Double_t rdMaxSigEtEff = 0;
Double_t sMaxBkgEtRate = 0;
Double_t rsMaxBkgEtRate = 0;
Double_t dMaxBkgEtRate = 0;
Double_t rdMaxBkgEtRate = 0;


Double_t maxSigEtEffSingle = 0;
Double_t minBkgEtRateSingle = 0;
Double_t maxBkgEtRateSingle = 0;
Double_t minSigEtEffDouble = 0;
Double_t maxSigEtEffDouble = 0;
Double_t minBkgEtRateDouble = 0;
Double_t maxBkgEtRateDouble = 0;
ostringstream os;

string sCut, rsCut, dCut, rdCut;
sCut = "ElecHLTCutVarsPreTracks_hltCutVars_SingleElecsPT_EGAMMAHLT.obj.Et > -999.";
rsCut = "ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.Et > -999.";
dCut = "ElecHLTCutVarsPreTracks_hltCutVars_DoubleElecsPT_EGAMMAHLT.obj.Et > -999.";
rdCut = "ElecHLTCutVarsPreTracks_hltCutVars_RelaxedDoubleElecsPT_EGAMMAHLT.obj.Et > -999.";
Long64_t sSigTotal = sigEvents->GetEntries(sCut.c_str());
Long64_t rsSigTotal = sigEvents->GetEntries(rsCut.c_str());
Long64_t dSigTotal = sigEvents->GetEntries(dCut.c_str());
Long64_t rdSigTotal = sigEvents->GetEntries(rdCut.c_str());
Long64_t sBkgTotal = bkgEvents->GetEntries(sCut.c_str());
Long64_t rsBkgTotal = bkgEvents->GetEntries(sCut.c_str());
Long64_t dBkgTotal = bkgEvents->GetEntries(sCut.c_str());
Long64_t rdBkgTotal = bkgEvents->GetEntries(sCut.c_str());
Double_t bkgTotal = bkgEvents->GetEntries();
Double_t lumi = 2.0E33;
Double_t xSection = 2.16E-2;
Double_t conversion = 1.0E-27;
Int_t i = 0;
Double_t sEtCut = 0.;
Double_t rsEtCut = 0.;
Double_t dEtCut = 0.;
Double_t rdEtCut = 0.;
for (i = 0; i < 120; i++) {
  sEtCut = (Double_t)i / 2.;
  os.str("");
  os<<"ElecHLTCutVarsPreTracks_hltCutVars_SingleElecsPT_EGAMMAHLT.obj.l1Match && ElecHLTCutVarsPreTracks_hltCutVars_SingleElecsPT_EGAMMAHLT.obj.Et > "<<sEtCut;
  sCut = os.str();

  rsEtCut = (Double_t)i / 2.;
  os.str("");
  os<<"ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.l1Match && ElecHLTCutVarsPreTracks_hltCutVars_RelaxedSingleElecsPT_EGAMMAHLT.obj.Et > "<<rsEtCut;
  rsCut = os.str();

  dEtCut = (Double_t)i / 2.;
  os.str("");
  os<<"ElecHLTCutVarsPreTracks_hltCutVars_DoubleElecsPT_EGAMMAHLT.obj.l1Match && ElecHLTCutVarsPreTracks_hltCutVars_DoubleElecsPT_EGAMMAHLT.obj.Et > "<<dEtCut;
  dCut = os.str();

  rdEtCut = (Double_t)i / 2.;
  os.str("");
  os<<"ElecHLTCutVarsPreTracks_hltCutVars_RelaxedDoubleElecsPT_EGAMMAHLT.obj.l1Match && ElecHLTCutVarsPreTracks_hltCutVars_RelaxedDoubleElecsPT_EGAMMAHLT.obj.Et > "<<rdEtCut;
  rdCut = os.str();

  sSigEt[i] = sigEvents->GetEntries(sCut.c_str());
  rsSigEt[i] = sigEvents->GetEntries(rsCut.c_str());
  dSigEt[i] = GetEntriesMulti(dCut.c_str(), sigEvents, 2);
  rdSigEt[i] = GetEntriesMulti(rdCut.c_str(), sigEvents, 2);
  sBkgEt[i] = bkgEvents->GetEntries(sCut.c_str());
  rsBkgEt[i] = bkgEvents->GetEntries(rsCut.c_str());
  dBkgEt[i] = GetEntriesMulti(dCut.c_str(), bkgEvents, 2);
  rdBkgEt[i] = GetEntriesMulti(rdCut.c_str(), bkgEvents, 2);

  if (sSigTotal != 0) sSigEtEff[i] = (Double_t)sSigEt[i] / (Double_t)sSigTotal;
  else sSigEtEff[i] = 0;
  if (rsSigTotal != 0) rsSigEtEff[i] = (Double_t)rsSigEt[i] / (Double_t)rsSigTotal;
  else rsSigEtEff[i] = 0;
  if (dSigTotal != 0) dSigEtEff[i] = (Double_t)dSigEt[i] / (Double_t)dSigTotal;
  else dSigEtEff[i] = 0;
  if (rdSigTotal != 0) rdSigEtEff[i] = (Double_t)rdSigEt[i] / (Double_t)rdSigTotal;
  else rdSigEtEff[i] = 0;
  if (sBkgTotal != 0) sBkgEtRate[i] = (Double_t)sBkgEt[i] / (Double_t)sBkgTotal * lumi * xSection * conversion;
  else sBkgEtRate[i] = 0;
  if (rsBkgTotal != 0) rsBkgEtRate[i] = (Double_t)rsBkgEt[i] / (Double_t)rsBkgTotal * lumi * xSection * conversion;
  else rsBkgEtRate[i] = 0;
  if (dBkgTotal != 0) dBkgEtRate[i] = (Double_t)dBkgEt[i] / (Double_t)dBkgTotal * lumi * xSection * conversion;
  else dBkgEtRate[i] = 0;
  if (rdBkgTotal != 0) rdBkgEtRate[i] = (Double_t)rdBkgEt[i] / (Double_t)rdBkgTotal * lumi * xSection * conversion;
  else rdBkgEtRate[i] = 0;

  if (i == 0) {
    sMinSigEtEff = sSigEtEff[i];
    rsMinSigEtEff = rsSigEtEff[i];
    dMinSigEtEff = dSigEtEff[i];
    rdMinSigEtEff = rdSigEtEff[i];
 
    sMinBkgEtRate = sBkgEtRate[i];
    rsMinBkgEtRate = rsBkgEtRate[i];
    dMinBkgEtRate = dBkgEtRate[i];
    rdMinBkgEtRate = rdBkgEtRate[i];

    sMaxSigEtEff = sSigEtEff[i];
    rsMaxSigEtEff = rsSigEtEff[i];
    dMaxSigEtEff = dSigEtEff[i];
    rdMaxSigEtEff = rdSigEtEff[i];
 
    sMaxBkgEtRate = sBkgEtRate[i];
    rsMaxBkgEtRate = rsBkgEtRate[i];
    dMaxBkgEtRate = dBkgEtRate[i];
    rdMaxBkgEtRate = rdBkgEtRate[i];
  }
  else {
    if (sSigEtEff[i] < sMinSigEtEff) sMinSigEtEff = sSigEtEff[i];
    if (rsSigEtEff[i] < rsMinSigEtEff) rsMinSigEtEff = rsSigEtEff[i];
    if (dSigEtEff[i] < dMinSigEtEff) dMinSigEtEff = dSigEtEff[i];
    if (rdSigEtEff[i] < rdMinSigEtEff) rdMinSigEtEff = rdSigEtEff[i];
    if (sBkgEtRate[i] < sMinBkgEtRate) sMinBkgEtRate = sBkgEtRate[i];
    if (rsBkgEtRate[i] < rsMinBkgEtRate) rsMinBkgEtRate = rsBkgEtRate[i];
    if (dBkgEtRate[i] < dMinBkgEtRate) dMinBkgEtRate = dBkgEtRate[i];
    if (rdBkgEtRate[i] < rdMinBkgEtRate) rdMinBkgEtRate = rdBkgEtRate[i];
    if (sSigEtEff[i] > sMaxSigEtEff) sMaxSigEtEff = sSigEtEff[i];
    if (rsSigEtEff[i] > rsMaxSigEtEff) rsMaxSigEtEff = rsSigEtEff[i];
    if (dSigEtEff[i] > dMaxSigEtEff) dMaxSigEtEff = dSigEtEff[i];
    if (rdSigEtEff[i] > rdMaxSigEtEff) rdMaxSigEtEff = rdSigEtEff[i];
    if (sBkgEtRate[i] > sMaxBkgEtRate) sMaxBkgEtRate = sBkgEtRate[i];
    if (rsBkgEtRate[i] > rsMaxBkgEtRate) rsMaxBkgEtRate = rsBkgEtRate[i];
    if (dBkgEtRate[i] > dMaxBkgEtRate) dMaxBkgEtRate = dBkgEtRate[i];
    if (rdBkgEtRate[i] > rdMaxBkgEtRate) rdMaxBkgEtRate = rdBkgEtRate[i];
  }
}
Double_t sSigEtEffRange = sMaxSigEtEff - sMinSigEtEff;
Double_t rsSigEtEffRange = rsMaxSigEtEff - rsMinSigEtEff;
Double_t dSigEtEffRange = dMaxSigEtEff - dMinSigEtEff;
Double_t rdSigEtEffRange = rdMaxSigEtEff - rdMinSigEtEff;
Double_t sBkgEtRateRange = sMaxBkgEtRate - sMinBkgEtRate;
Double_t rsBkgEtRateRange = rsMaxBkgEtRate - rsMinBkgEtRate;
Double_t dBkgEtRateRange = dMaxBkgEtRate - dMinBkgEtRate;
Double_t rdBkgEtRateRange = rdMaxBkgEtRate - rdMinBkgEtRate;

if (sSigEtEffRange == 0) sMaxSigEtEff = sMinSigEtEff + 1.;
if (rsSigEtEffRange == 0) rsMaxSigEtEff = rsMinSigEtEff + 1.;
if (dSigEtEffRange == 0) dMaxSigEtEff = dMinSigEtEff + 1.;
if (rdSigEtEffRange == 0) rdMaxSigEtEff = rdMinSigEtEff + 1.;
if (sBkgEtRateRange == 0) sMaxBkgEtRate = sMinBkgEtRate + 1.;
if (rsBkgEtRateRange == 0) rsMaxBkgEtRate = rsMinBkgEtRate + 1.;
if (dBkgEtRateRange == 0) dMaxBkgEtRate = dMinBkgEtRate + 1.;
if (rdBkgEtRateRange == 0) rdMaxBkgEtRate = rdMinBkgEtRate + 1.;

TH2F *sEffVBkg = new TH2F("sEffVBkg", "Efficiency vs. Background in Single Electron Stream", 1000, sMinBkgEtRate - 0.1*sBkgEtRateRange, sMaxBkgEtRate + 0.1*sBkgEtRateRange, 1000, sMinSigEtEff - 0.1*sSigEtEffRange, sMaxSigEtEff + 0.1*sSigEtEffRange);
TH2F *rsEffVBkg = new TH2F("rsEffVBkg", "Efficiency vs. Background in Relaxed Single Electron Stream", 1000, rsMinBkgEtRate - 0.1*rsBkgEtRateRange, rsMaxBkgEtRate + 0.1*rsBkgEtRateRange, 1000, rsMinSigEtEff - 0.1*rsSigEtEffRange, rsMaxSigEtEff + 0.1*rsSigEtEffRange);
TH2F *dEffVBkg = new TH2F("dEffVBkg", "Efficiency vs. Background in Double Electron Stream", 1000, dMinBkgEtRate - 0.1*dBkgEtRateRange, dMaxBkgEtRate + 0.1*dBkgEtRateRange, 1000, dMinSigEtEff - 0.1*dSigEtEffRange, dMaxSigEtEff + 0.1*dSigEtEffRange);
TH2F *rdEffVBkg = new TH2F("rdEffVBkg", "Efficiency vs. Background in Relaxed Double Electron Stream", 1000, rdMinBkgEtRate - 0.1*rdBkgEtRateRange, rdMaxBkgEtRate + 0.1*rdBkgEtRateRange, 1000, rdMinSigEtEff - 0.1*rdSigEtEffRange, rdMaxSigEtEff + 0.1*rdSigEtEffRange);
for (Int_t i = 0; i < 120; i++) {
  sEffVBkg->Fill(sBkgEtRate[i], sSigEtEff[i]);
  rsEffVBkg->Fill(rsBkgEtRate[i], rsSigEtEff[i]);
  dEffVBkg->Fill(dBkgEtRate[i], dSigEtEff[i]);
  rdEffVBkg->Fill(rdBkgEtRate[i], rdSigEtEff[i]);
}
gStyle->SetOptStat(0000000);                                
TCanvas *myCanvas = new TCanvas("myCanvas", "Efficiency vs. Background for Et Cut", 1000, 1000);
myCanvas->Divide(2,2);
myCanvas->cd(1);
sEffVBkg->Draw();
myCanvas->cd(2);
rsEffVBkg->Draw();
myCanvas->cd(3);
dEffVBkg->Draw();
myCanvas->cd(4);
rdEffVBkg->Draw();
}
