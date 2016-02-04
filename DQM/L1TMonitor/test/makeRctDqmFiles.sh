#!/bin/bash

########## Files created by this script ##########
#
#        L1TRCToffline_cff.py  in /python/
# Rct_LUTconfiguration_cff.py  in /python/
#           testRCToffline.py  in /test/
#                   lsfJob.csh in /test/
#                      rct.C   in /test/
#
##################################################

nArguments=3
badArgumentExit=64

if [ $# -ne "$nArguments" ]
then
  echo "Syntax: ./makeRctDqmPlots.sh key nEvents run"
  echo "CRAFT 1 keys:"
  echo "  EEG_EHSUMS_TAU3_DECO_25_CRAFT1"
  echo "  EEG_EHSUMS_TAU3_DECO_25_CRAFT1_NOHF"
  echo "  EEG_ESUMS_TAU3_DECOMPRESSED_25_V4_1"
  echo "  HEG_HSUMS_HF"
  echo "  HEG_HSUMS_TAU3_DECO_25_CRAFT1"
  echo "  HEG_HSUMS_TAU3_DECO_25_CRAFT1_NOHF"
  exit $badArgumentExit
fi

key=$1
nEvents=$2
run=$3

########## Begin CAF configuration ##########
global=Commissioning08
path=Calo/RAW/v1
nFiles=55
########## End CAF configuration ##########

a=0${run:0:2}
b=${run:2}

if [ ! -d $CMSSW_BASE/src/DQM/L1TMonitor/test/run$run ]
then
  mkdir $CMSSW_BASE/src/DQM/L1TMonitor/test/run$run
fi

cat << EOF > "$CMSSW_BASE/src/DQM/L1TMonitor/python/L1TRCToffline_${run}_cff.py"
import FWCore.ParameterSet.Config as cms

# emap_from_ascii = cms.ESSource("HcalTextCalibrations",
#     input = cms.VPSet(cms.PSet(
#         object = cms.string('ElectronicsMap'),
#         file = cms.FileInPath('official_emap_v5_080208.txt.new_trig')
#     ))
# )
#es_prefer = cms.ESPrefer("HcalTextCalibrations","emap_from_ascii")

#global configuration
from Configuration.StandardSequences.FrontierConditions_GlobalTag_cff import *
es_prefer_GlobalTag = cms.ESPrefer("PoolDBESSource","GlobalTag")
#off-line
GlobalTag.globaltag = 'CRUZET4_V1::All'
GlobalTag.connect = 'frontier://FrontierProd/CMS_COND_21X_GLOBALTAG'
#on-line
#GlobalTag.globaltag = 'CRZT210_V1H::All'
#GlobalTag.connect = 'frontier://(proxyurl=http://localhost:3128)(serverurl=http://frontier1.cms:8000/FrontierOnProd)(serverurl=http://frontier2.cms:8000/FrontierOnProd)(retrieve-ziplevel=0)/CMS_COND_21X_GLOBALTAG'


#unpacking
from Configuration.StandardSequences.RawToDigi_Data_cff import *

#emulator/comparator
from L1Trigger.HardwareValidation.L1HardwareValidation_cff import *
from L1Trigger.Configuration.L1Config_cff import *

#for LUTs
from DQM.L1TMonitor.Rct_LUTconfiguration_${run}_cff import *

# For the GT
from L1TriggerConfig.L1GtConfigProducers.L1GtConfig_cff import *

#dqm
rctEmulDigis = cms.EDProducer("L1RCTProducer",
    hcalDigisLabel = cms.InputTag("hcalTriggerPrimitiveDigis"),
    useDebugTpgScales = cms.bool(False),
    useEcalCosmicTiming = cms.bool(False),
    postSamples = cms.uint32(0),
    preSamples = cms.uint32(0),
    useHcalCosmicTiming = cms.bool(False),
    useEcal = cms.bool(True),
    useHcal = cms.bool(True),
    ecalDigisLabel = cms.InputTag("ecalTriggerPrimitiveDigis"),
    UseMCAsInput = cms.bool(False),
    HFShift = cms.int32(-2),
    HBShift = cms.int32(1),
    useCorrectionsLindsey = cms.bool(False)
)

rctEmulDigis.hcalDigisLabel='hcalDigis'
#rctEmulDigis.ecalDigisLabel='ecalEBunpacker'
rctEmulDigis.ecalDigisLabel='ecalDigis:EcalTriggerPrimitives'

l1tderct = cms.EDFilter("L1TdeRCT",
    rctSourceData = cms.InputTag("l1GctHwDigis"),
    HistFolder = cms.untracked.string('L1TEMU/L1TdeRCT/'),
    outputFile = cms.untracked.string('./run$run.root'),
    verbose = cms.untracked.bool(False),
    DQMStore = cms.untracked.bool(True),
    singlechannelhistos = cms.untracked.bool(True),
    ecalTPGData = cms.InputTag("",""),
    rctSourceEmul = cms.InputTag("rctDigis"),
    disableROOToutput = cms.untracked.bool(False),
    hcalTPGData = cms.InputTag(""),
    gtEGAlgoName = cms.string("L1_SingleEG1"),
    doubleThreshold = cms.int32(3),
    gtDigisLabel = cms.InputTag("gtDigis")
)

l1tderct.rctSourceData = 'gctDigis'
l1tderct.rctSourceEmul = 'rctEmulDigis'
#l1tderct.ecalTPGData = 'ecalEBunpacker:EcalTriggerPrimitives'
l1tderct.ecalTPGData = 'ecalDigis:EcalTriggerPrimitives'
l1tderct.hcalTPGData = 'hcalDigis'

l1trct = cms.EDFilter("L1TRCT",
    DQMStore = cms.untracked.bool(True),
    disableROOToutput = cms.untracked.bool(False),
    outputFile = cms.untracked.string('./run$run.root'),
    rctSource = cms.InputTag("l1GctHwDigis","","DQM"),
    verbose = cms.untracked.bool(False)
)

l1trct.rctSource = 'gctDigis'

p = cms.Path(
    cms.SequencePlaceholder("RawToDigi")
    *cms.SequencePlaceholder("rctEmulDigis")
    *cms.SequencePlaceholder("l1trct")
    *cms.SequencePlaceholder("l1tderct")
    )
EOF

cat $CMSSW_BASE/src/${key}_cff.py > "$CMSSW_BASE/src/DQM/L1TMonitor/python/Rct_LUTconfiguration_${run}_cff.py"

cat << EOF > "$CMSSW_BASE/src/DQM/L1TMonitor/test/testRCToffline_${run}.py"
import FWCore.ParameterSet.Config as cms

process = cms.Process("RCTofflineTEST")

#process.load("DQMServices.Core.DQM_cfg")
process.DQMStore = cms.Service("DQMStore")

process.load("DQM/L1TMonitor/L1TRCToffline_${run}_cff")

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32($nEvents)
)

process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
EOF

nsls /castor/cern.ch/cms/store/data/$global/$path/000/$a/$b |
head -n $nFiles |
awk -v globalTemp=$global -v pathTemp=$path -v aTemp=$a -v bTemp=$b '{print "        '\''rfio:/castor/cern.ch/cms/store/data/" globalTemp "/" pathTemp "/000/" aTemp "/" bTemp "/" $1 "'\'',"}' |
sed '$s/.$//' >> "$CMSSW_BASE/src/DQM/L1TMonitor/test/testRCToffline_${run}.py"

cat << EOF >> "$CMSSW_BASE/src/DQM/L1TMonitor/test/testRCToffline_${run}.py"
    )
)
EOF

cat << EOF > "$CMSSW_BASE/src/DQM/L1TMonitor/test/lsfJob.csh"
#!/bin/csh
cd $CMSSW_BASE/src/DQM/L1TMonitor/test
eval `scramv1 runtime -csh`
cmsRun testRCToffline_${run}.py
EOF

chmod +x lsfJob.csh

if [ ! -f $CMSSW_BASE/src/DQM/L1TMonitor/test/rct.C ]
then
  cat << EOF > "$CMSSW_BASE/src/DQM/L1TMonitor/test/rct.C"
Int_t paletteSize;
Int_t nContours;

Int_t pEff    [199];
Int_t pIneff  [199];
Int_t pOvereff[199];

TFile* f;

TCanvas* c1;

TH2F* dummybox;

TString runNumber = 0;

void setTDRStyle() {
  TStyle *tdrStyle = new TStyle("tdrStyle","Style for P-TDR");

// For the canvas:
  tdrStyle->SetCanvasBorderMode(0);
  tdrStyle->SetCanvasColor(kWhite);
  tdrStyle->SetCanvasDefH(540); //Height of canvas
//   tdrStyle->SetCanvasDefW(600); //Width of canvas
  tdrStyle->SetCanvasDefW(632); //Width of canvas
  tdrStyle->SetCanvasDefX(0);   //POsition on screen
  tdrStyle->SetCanvasDefY(0);

// For the Pad:
  tdrStyle->SetPadBorderMode(0);
  // tdrStyle->SetPadBorderSize(Width_t size = 1);
  tdrStyle->SetPadColor(kWhite);
  tdrStyle->SetPadGridX(false);
  tdrStyle->SetPadGridY(false);
  tdrStyle->SetGridColor(0);
  tdrStyle->SetGridStyle(3);
  tdrStyle->SetGridWidth(1);

// For the frame:
  tdrStyle->SetFrameBorderMode(0);
  tdrStyle->SetFrameBorderSize(1);
  tdrStyle->SetFrameFillColor(0);
  tdrStyle->SetFrameFillStyle(0);
  tdrStyle->SetFrameLineColor(1);
  tdrStyle->SetFrameLineStyle(1);
  tdrStyle->SetFrameLineWidth(1);

// For the histo:
  // tdrStyle->SetHistFillColor(1);
  // tdrStyle->SetHistFillStyle(0);
  tdrStyle->SetHistLineColor(1);
  tdrStyle->SetHistLineStyle(0);
  tdrStyle->SetHistLineWidth(1);
  // tdrStyle->SetLegoInnerR(Float_t rad = 0.5);
  // tdrStyle->SetNumberContours(Int_t number = 20);

  tdrStyle->SetEndErrorSize(2);
  // tdrStyle->SetErrorMarker(20);
  tdrStyle->SetErrorX(0.);
  
  tdrStyle->SetMarkerStyle(20);

//For the fit/function:
  tdrStyle->SetOptFit(1);
  tdrStyle->SetFitFormat("5.4g");
  tdrStyle->SetFuncColor(2);
  tdrStyle->SetFuncStyle(1);
  tdrStyle->SetFuncWidth(1);

//For the date:
  tdrStyle->SetOptDate(0);
  // tdrStyle->SetDateX(Float_t x = 0.01);
  // tdrStyle->SetDateY(Float_t y = 0.01);

// For the statistics box:
  tdrStyle->SetOptFile(0);
  tdrStyle->SetOptStat(0); // To display the mean and RMS:   SetOptStat("mr");
  tdrStyle->SetStatColor(kWhite);
  tdrStyle->SetStatFont(42);
  tdrStyle->SetStatFontSize(0.04);
  tdrStyle->SetStatTextColor(1);
  tdrStyle->SetStatFormat("6.4g");
  tdrStyle->SetStatBorderSize(1);
  tdrStyle->SetStatH(0.1);
  tdrStyle->SetStatW(0.2);
  // tdrStyle->SetStatStyle(Style_t style = 1001);
  // tdrStyle->SetStatX(Float_t x = 0);
  // tdrStyle->SetStatY(Float_t y = 0);

// Margins:
  tdrStyle->SetPadTopMargin(0.1);
  tdrStyle->SetPadBottomMargin(0.1);
  tdrStyle->SetPadLeftMargin(0.1);
//   tdrStyle->SetPadRightMargin(0.02);
  tdrStyle->SetPadRightMargin(0.1);

// For the Global title:

  tdrStyle->SetOptTitle(1);
  tdrStyle->SetTitleFont(42);
  tdrStyle->SetTitleColor(1);
  tdrStyle->SetTitleTextColor(1);
  tdrStyle->SetTitleFillColor(10);
  tdrStyle->SetTitleFontSize(0.05);
  // tdrStyle->SetTitleH(0); // Set the height of the title box
  // tdrStyle->SetTitleW(0); // Set the width of the title box
  // tdrStyle->SetTitleX(0); // Set the position of the title box
  // tdrStyle->SetTitleY(0.985); // Set the position of the title box
  // tdrStyle->SetTitleStyle(Style_t style = 1001);
  // tdrStyle->SetTitleBorderSize(2);

// For the axis titles:

  tdrStyle->SetTitleColor(1, "XYZ");
  tdrStyle->SetTitleFont(42, "XYZ");
  tdrStyle->SetTitleSize(0.06, "XYZ");
  // tdrStyle->SetTitleXSize(Float_t size = 0.02); // Another way to set the size?
  // tdrStyle->SetTitleYSize(Float_t size = 0.02);
  tdrStyle->SetTitleXOffset(0.9);
  tdrStyle->SetTitleYOffset(1.25);
  // tdrStyle->SetTitleOffset(1.1, "Y"); // Another way to set the Offset

// For the axis labels:

  tdrStyle->SetLabelColor(1, "XYZ");
  tdrStyle->SetLabelFont(42, "XYZ");
  tdrStyle->SetLabelOffset(0.007, "XYZ");
  tdrStyle->SetLabelSize(0.05, "XYZ");

// For the axis:

  tdrStyle->SetAxisColor(1, "XYZ");
  tdrStyle->SetStripDecimals(kTRUE);
  tdrStyle->SetTickLength(0.03, "XYZ");
  tdrStyle->SetNdivisions(510, "XYZ");
  tdrStyle->SetPadTickX(1);  // To get tick marks on the opposite side of the frame
  tdrStyle->SetPadTickY(1);

// Change for log plots:
  tdrStyle->SetOptLogx(0);
  tdrStyle->SetOptLogy(0);
  tdrStyle->SetOptLogz(0);

// Postscript options:
  tdrStyle->SetPaperSize(20.,20.);
  // tdrStyle->SetLineScalePS(Float_t scale = 3);
  // tdrStyle->SetLineStyleString(Int_t i, const char* text);
  // tdrStyle->SetHeaderPS(const char* header);
  // tdrStyle->SetTitlePS(const char* pstitle);

  // tdrStyle->SetBarOffset(Float_t baroff = 0.5);
  // tdrStyle->SetBarWidth(Float_t barwidth = 0.5);
  // tdrStyle->SetPaintTextFormat(const char* format = "g");
  // tdrStyle->SetPalette(Int_t ncolors = 0, Int_t* colors = 0);
  // tdrStyle->SetTimeOffset(Double_t toffset);
  // tdrStyle->SetHistMinimumZero(kTRUE);

  tdrStyle->cd();

}

void init() {
  Float_t rgb[597] = {0};

  paletteSize = 199;
  nContours   = 199;

  f = new TFile("run" + runNumber + ".root");

  c1 = new TCanvas;

  dummybox = new TH2F("dummy", "; GCT #eta; GCT #phi", 22, -0.5, 21.5, 18, -0.5, 17.5);

  for (Int_t i = 0; i < paletteSize; i++) {
    rgb[3 * i + 0] = 1.0;
    rgb[3 * i + 1] = 0.0;
    rgb[3 * i + 2] = 0.0;
    
    if (i >= 169) {
      rgb[3 * i + 0] = 1.0;
      rgb[3 * i + 1] = 0.5;
      rgb[3 * i + 2] = 0.0;
    }

    if (i >= 179) {
      rgb[3 * i + 0] = 1.0;
      rgb[3 * i + 1] = 1.0;
      rgb[3 * i + 2] = 0.0;
    }

    if (i >= 189) {
      rgb[3 * i + 0] = 0.5;
      rgb[3 * i + 1] = 1.0;
      rgb[3 * i + 2] = 0.0;
    }

    if (i >= 197) {
      rgb[3 * i + 0] = 0.0;
      rgb[3 * i + 1] = 0.8;
      rgb[3 * i + 2] = 0.0;
    }

    pEff [i] = TColor::GetColor (rgb[3 * i + 0], rgb[3 * i + 1], rgb[3 * i + 2]);
  }

  for (Int_t i = 0; i < paletteSize; i++) {
    rgb[3 * i + 0] = 1.0;
    rgb[3 * i + 1] = 0.0;
    rgb[3 * i + 2] = 0.0;

    if (i <= 28) {
      rgb[3 * i + 0] = 1.0;
      rgb[3 * i + 1] = 0.5;
      rgb[3 * i + 2] = 0.0;
    }

    if (i <= 18) {
      rgb[3 * i + 0] = 1.0;
      rgb[3 * i + 1] = 1.0;
      rgb[3 * i + 2] = 0.0;
    }

    if (i <= 8) {
      rgb[3 * i + 0] = 0.5;
      rgb[3 * i + 1] = 1.0;
      rgb[3 * i + 2] = 0.0;
    }

    if (i <= 1) {
      rgb[3 * i + 0] = 0.0;
      rgb[3 * i + 1] = 0.8;
      rgb[3 * i + 2] = 0.0;
    }

    pIneff [i] = TColor::GetColor (rgb[3 * i + 0], rgb[3 * i + 1], rgb[3 * i + 2]);
  }

  for (Int_t i = 0; i < paletteSize; i++) {
    rgb[3 * i + 0] = 1.0;
    rgb[3 * i + 1] = 0.0;
    rgb[3 * i + 2] = 0.0;
    
    if (i <= 28) {
      rgb[3 * i + 0] = 1.0;
      rgb[3 * i + 1] = 0.5;
      rgb[3 * i + 2] = 0.0;
    }

    if (i <= 18) {
      rgb[3 * i + 0] = 1.0;
      rgb[3 * i + 1] = 1.0;
      rgb[3 * i + 2] = 0.0;
    }

    if (i <= 8) {
      rgb[3 * i + 0] = 0.5;
      rgb[3 * i + 1] = 1.0;
      rgb[3 * i + 2] = 0.0;
    }

    if (i <= 1) {
      rgb[3 * i + 0] = 0.0;
      rgb[3 * i + 1] = 0.8;
      rgb[3 * i + 2] = 0.0;
    }

    pOvereff [i] = TColor::GetColor (rgb[3 * i + 0], rgb[3 * i + 1], rgb[3 * i + 2]);
  }

  for (int i = 0; i < 22; i++)
    for (int j = 0; j < 18; j++)
      dummybox->Fill (i, j);
}

void loop(TString runString) {
  runNumber = runString;

  init();

  f->cd("run" + runNumber + ".root:/DQMData/L1TEMU/L1TdeRCT");

  gStyle->SetOptStat("e");

  rctInputTPGEcalOcc->SetTitle("ECAL TPG occupancy");
  rctInputTPGEcalOcc->SetXTitle("GCT #eta");
  rctInputTPGEcalOcc->SetYTitle("GCT #phi");
  rctInputTPGEcalOcc->Draw("box");
  c1->SaveAs("./run" + runNumber + "/ecalTpgOcc.png");

  rctInputTPGHcalOcc->SetTitle("HCAL TPG occupancy");
  rctInputTPGHcalOcc->SetXTitle("GCT #eta");
  rctInputTPGHcalOcc->SetYTitle("GCT #phi");
  rctInputTPGHcalOcc->Draw("box");
  c1->SaveAs("./run" + runNumber + "/hcalTpgOcc.png");

  f->cd("run" + runNumber + ".root:/DQMData/L1TEMU/L1TdeRCT/BitData/ServiceData");

  rctBitDataHfPlusTau2D->SetTitle("HF + tau bit occupancy from data");
  rctBitDataHfPlusTau2D->SetXTitle("GCT #eta");
  rctBitDataHfPlusTau2D->SetYTitle("GCT #phi");
  rctBitDataHfPlusTau2D->Draw("box");
  c1->SaveAs("./run" + runNumber + "/hfPlusTauOccData.png");
  
  rctBitDataMip2D->SetTitle("MIP bit occupancy from data");
  rctBitDataMip2D->SetXTitle("GCT #eta");
  rctBitDataMip2D->SetYTitle("GCT #phi");
  rctBitDataMip2D->Draw("box");
  c1->SaveAs("./run" + runNumber + "/mipOccData.png");
  
  rctBitDataOverFlow2D->SetTitle("Overflow bit occupancy from data");
  rctBitDataOverFlow2D->SetXTitle("GCT #eta");
  rctBitDataOverFlow2D->SetYTitle("GCT #phi");
  rctBitDataOverFlow2D->Draw("box");
  c1->SaveAs("./run" + runNumber + "/overFlowOccData.png");
  
  rctBitDataQuiet2D->SetTitle("Quiet bit occupancy from data");
  rctBitDataQuiet2D->SetXTitle("GCT #eta");
  rctBitDataQuiet2D->SetYTitle("GCT #phi");
  rctBitDataQuiet2D->Draw("box");
  c1->SaveAs("./run" + runNumber + "/quietOccData.png");
  
//  rctBitDataTauVeto2D->SetTitle("Tau veto bit occupancy from data");
//  rctBitDataTauVeto2D->SetXTitle("GCT #eta");
//  rctBitDataTauVeto2D->SetYTitle("GCT #phi");
//  rctBitDataTauVeto2D->Draw("box");
//  c1->SaveAs("./run" + runNumber + "/tauVetoOccData.png");
  
  rctBitEmulHfPlusTau2D->SetTitle("HF + tau bit occupancy from emulator");
  rctBitEmulHfPlusTau2D->SetXTitle("GCT #eta");
  rctBitEmulHfPlusTau2D->SetYTitle("GCT #phi");
  rctBitEmulHfPlusTau2D->Draw("box");
  c1->SaveAs("./run" + runNumber + "/hfPlusTauOccEmul.png");
  
  rctBitEmulMip2D->SetTitle("MIP bit occupancy from emulator");
  rctBitEmulMip2D->SetXTitle("GCT #eta");
  rctBitEmulMip2D->SetYTitle("GCT #phi");
  rctBitEmulMip2D->Draw("box");
  c1->SaveAs("./run" + runNumber + "/mipOccEmul.png");
  
  rctBitEmulOverFlow2D->SetTitle("Overflow bit occupancy from emulator");
  rctBitEmulOverFlow2D->SetXTitle("GCT #eta");
  rctBitEmulOverFlow2D->SetYTitle("GCT #phi");
  rctBitEmulOverFlow2D->Draw("box");
  c1->SaveAs("./run" + runNumber + "/overFlowOccEmul.png");
  
  rctBitEmulQuiet2D->SetTitle("Quiet bit occupancy from emulator");
  rctBitEmulQuiet2D->SetXTitle("GCT #eta");
  rctBitEmulQuiet2D->SetYTitle("GCT #phi");
  rctBitEmulQuiet2D->Draw("box");
  c1->SaveAs("./run" + runNumber + "/quietOccEmul.png");
  
//  rctBitEmulTauVeto2D->SetTitle("Tau veto bit occupancy from emulator");
//  rctBitEmulTauVeto2D->SetXTitle("GCT #eta");
//  rctBitEmulTauVeto2D->SetYTitle("GCT #phi");
//  rctBitEmulTauVeto2D->Draw("box");
//  c1->SaveAs("./run" + runNumber + "/tauVetoOccEmul.png");
  
  f->cd("run" + runNumber + ".root:/DQMData/L1TEMU/L1TdeRCT/BitData");

  gStyle->SetNumberContours (nContours);
  gStyle->SetOptStat(0);
  gStyle->SetPalette (paletteSize, pEff);
  
  rctBitHfPlusTauEff2D->SetTitle("HF + tau bit efficiency");
  rctBitHfPlusTauEff2D->SetXTitle("GCT #eta");
  rctBitHfPlusTauEff2D->SetYTitle("GCT #phi");
  rctBitHfPlusTauEff2D->SetMinimum(0.005);
  rctBitHfPlusTauEff2D->SetMaximum(1.0);
  rctBitHfPlusTauEff2D->Draw("colz");
  dummybox->Draw("box, same");
  c1->SaveAs("./run" + runNumber + "/hfPlusTauEff.png");
  
  rctBitMipEff2D->SetTitle("MIP bit efficiency");
  rctBitMipEff2D->SetXTitle("GCT #eta");
  rctBitMipEff2D->SetYTitle("GCT #phi");
  rctBitMipEff2D->SetMinimum(0.005);
  rctBitMipEff2D->SetMaximum(1.0);
  rctBitMipEff2D->Draw("colz");
  dummybox->Draw("box, same");
  c1->SaveAs("./run" + runNumber + "/mipEff.png");
  
  rctBitOverFlowEff2D->SetTitle("Overflow bit efficiency");
  rctBitOverFlowEff2D->SetXTitle("GCT #eta");
  rctBitOverFlowEff2D->SetYTitle("GCT #phi");
  rctBitOverFlowEff2D->SetMinimum(0.005);
  rctBitOverFlowEff2D->SetMaximum(1.0);
  rctBitOverFlowEff2D->Draw("colz");
  dummybox->Draw("box, same");
  c1->SaveAs("./run" + runNumber + "/overFlowEff.png");
  
  rctBitQuietEff2D->SetTitle("Quiet bit efficiency");
  rctBitQuietEff2D->SetXTitle("GCT #eta");
  rctBitQuietEff2D->SetYTitle("GCT #phi");
  rctBitQuietEff2D->SetMinimum(0.005);
  rctBitQuietEff2D->SetMaximum(1.0);
  rctBitQuietEff2D->Draw("colz");
  dummybox->Draw("box, same");
  c1->SaveAs("./run" + runNumber + "/quietEff.png");
  
//  rctBitTauVetoEff2D->SetTitle("Tau veto bit efficiency");
//  rctBitTauVetoEff2D->SetXTitle("GCT #eta");
//  rctBitTauVetoEff2D->SetYTitle("GCT #phi");
//  rctBitTauVetoEff2D->Draw("colz");
//  dummybox->Draw("box, same");
//  c1->SaveAs("./run" + runNumber + "/tauVetoEff.png");
  
  gStyle->SetPalette (paletteSize, pIneff);
  
  rctBitHfPlusTauIneff2D->SetTitle("HF + tau bit inefficiency");
  rctBitHfPlusTauIneff2D->SetXTitle("GCT #eta");
  rctBitHfPlusTauIneff2D->SetYTitle("GCT #phi");
  rctBitHfPlusTauIneff2D->SetMinimum(0.005);
  rctBitHfPlusTauIneff2D->SetMaximum(1.0);
  rctBitHfPlusTauIneff2D->Draw("colz");
  dummybox->Draw("box, same");
  c1->SaveAs("./run" + runNumber + "/hfPlusTauInEff.png");
  
  rctBitMipIneff2D->SetTitle("MIP bit inefficiency");
  rctBitMipIneff2D->SetXTitle("GCT #eta");
  rctBitMipIneff2D->SetYTitle("GCT #phi");
  rctBitMipIneff2D->SetMinimum(0.005);
  rctBitMipIneff2D->SetMaximum(1.0);
  rctBitMipIneff2D->Draw("colz");
  dummybox->Draw("box, same");
  c1->SaveAs("./run" + runNumber + "/mipInEff.png");
  
  rctBitOverFlowIneff2D->SetTitle("Overflow bit inefficiency");
  rctBitOverFlowIneff2D->SetXTitle("GCT #eta");
  rctBitOverFlowIneff2D->SetYTitle("GCT #phi");
  rctBitOverFlowIneff2D->SetMinimum(0.005);
  rctBitOverFlowIneff2D->SetMaximum(1.0);
  rctBitOverFlowIneff2D->Draw("colz");
  dummybox->Draw("box, same");
  c1->SaveAs("./run" + runNumber + "/overFlowInEff.png");
  
  rctBitQuietIneff2D->SetTitle("Quiet bit inefficiency");
  rctBitQuietIneff2D->SetXTitle("GCT #eta");
  rctBitQuietIneff2D->SetYTitle("GCT #phi");
  rctBitQuietIneff2D->SetMinimum(0.005);
  rctBitQuietIneff2D->SetMaximum(1.0);
  rctBitQuietIneff2D->Draw("colz");
  dummybox->Draw("box, same");
  c1->SaveAs("./run" + runNumber + "/quietInEff.png");
  
//  rctBitTauVetoIneff2D->SetTitle("Tau veto bit inefficiency");
//  rctBitTauVetoIneff2D->SetXTitle("GCT #eta");
//  rctBitTauVetoIneff2D->SetYTitle("GCT #phi");
//  rctBitTauVetoIneff2D->SetMinimum(0.005);
//  rctBitTauVetoIneff2D->SetMaximum(1.0);
//  rctBitTauVetoIneff2D->Draw("colz");
//  dummybox->Draw("box, same");
//  c1->SaveAs("./run" + runNumber + "/tauVetoInEff.png");
  
  gStyle->SetPalette (paletteSize, pOvereff);
  
  rctBitHfPlusTauOvereff2D->SetTitle("HF + tau bit overefficiency");
  rctBitHfPlusTauOvereff2D->SetXTitle("GCT #eta");
  rctBitHfPlusTauOvereff2D->SetYTitle("GCT #phi");
  rctBitHfPlusTauOvereff2D->SetMinimum(0.005);
  rctBitHfPlusTauOvereff2D->SetMaximum(1.0);
  rctBitHfPlusTauOvereff2D->Draw("colz");
  dummybox->Draw("box, same");
  c1->SaveAs("./run" + runNumber + "/hfPlusTauOverEff.png");
  
  rctBitMipOvereff2D->SetTitle("MIP bit overefficiency");
  rctBitMipOvereff2D->SetXTitle("GCT #eta");
  rctBitMipOvereff2D->SetYTitle("GCT #phi");
  rctBitMipOvereff2D->SetMinimum(0.005);
  rctBitMipOvereff2D->SetMaximum(1.0);
  rctBitMipOvereff2D->Draw("colz");
  dummybox->Draw("box, same");
  c1->SaveAs("./run" + runNumber + "/mipOverEff.png");
  
  rctBitOverFlowOvereff2D->SetTitle("Overflow bit overefficiency");
  rctBitOverFlowOvereff2D->SetXTitle("GCT #eta");
  rctBitOverFlowOvereff2D->SetYTitle("GCT #phi");
  rctBitOverFlowOvereff2D->SetMinimum(0.005);
  rctBitOverFlowOvereff2D->SetMaximum(1.0);
  rctBitOverFlowOvereff2D->Draw("colz");
  dummybox->Draw("box, same");
  c1->SaveAs("./run" + runNumber + "/overFlowOverEff.png");
  
  rctBitQuietOvereff2D->SetTitle("Quiet bit overefficiency");
  rctBitQuietOvereff2D->SetXTitle("GCT #eta");
  rctBitQuietOvereff2D->SetYTitle("GCT #phi");
  rctBitQuietOvereff2D->SetMinimum(0.005);
  rctBitQuietOvereff2D->SetMaximum(1.0);
  rctBitQuietOvereff2D->Draw("colz");
  dummybox->Draw("box, same");
  c1->SaveAs("./run" + runNumber + "/quietOverEff.png");
  
//  rctBitTauVetoOvereff2D->SetTitle("Tau veto bit overefficiency");
//  rctBitTauVetoOvereff2D->SetXTitle("GCT #eta");
//  rctBitTauVetoOvereff2D->SetYTitle("GCT #phi");
//  rctBitTauVetoOvereff2D->SetMinimum(0.005);
//  rctBitTauVetoOvereff2D->SetMaximum(1.0);
//  rctBitTauVetoOvereff2D->Draw("colz");
//  dummybox->Draw("box, same");
//  c1->SaveAs("./run" + runNumber + "/tauVetoOverEff.png");
  
  f->cd("run" + runNumber + ".root:/DQMData/L1TEMU/L1TdeRCT/NisoEm/ServiceData");

  gStyle->SetOptStat("e");

  rctNisoEmDataOcc->SetTitle("Nonisolated electron occupancy from data");
  rctNisoEmDataOcc->SetXTitle("GCT #eta");
  rctNisoEmDataOcc->SetYTitle("GCT #phi");
  rctNisoEmDataOcc->Draw("box");
  c1->SaveAs("./run" + runNumber + "/nonIsoOccData.png");

  rctNisoEmEmulOcc->SetTitle("Nonisolated electron occupancy from emulator");
  rctNisoEmEmulOcc->SetXTitle("GCT #eta");
  rctNisoEmEmulOcc->SetYTitle("GCT #phi");
  rctNisoEmEmulOcc->Draw("box");
  c1->SaveAs("./run" + runNumber + "/nonIsoOccEmul.png");

  f->cd("run" + runNumber + ".root:/DQMData/L1TEMU/L1TdeRCT/NisoEm");

  gStyle->SetOptStat(0);
  gStyle->SetPalette (paletteSize, pEff);

  rctNisoEmEff1->SetTitle("Nonisolated electron efficiency 1");
  rctNisoEmEff1->SetXTitle("GCT #eta");
  rctNisoEmEff1->SetYTitle("GCT #phi");
  rctNisoEmEff1->SetMinimum(0.005);
  rctNisoEmEff1->SetMaximum(1.0);
  rctNisoEmEff1->Draw("colz");
  dummybox->Draw("box, same");
  c1->SaveAs("./run" + runNumber + "/nonIsoEff1.png");

  rctNisoEmEff2->SetTitle("Nonisolated electron efficiency 2");
  rctNisoEmEff2->SetXTitle("GCT #eta");
  rctNisoEmEff2->SetYTitle("GCT #phi");
  rctNisoEmEff2->SetMinimum(0.005);
  rctNisoEmEff2->SetMaximum(1.0);
  rctNisoEmEff2->Draw("colz");
  dummybox->Draw("box, same");
  c1->SaveAs("./run" + runNumber + "/nonIsoEff2.png");

  gStyle->SetPalette (paletteSize, pIneff);

  rctNisoEmIneff->SetTitle("Nonisolated electron inefficiency");
  rctNisoEmIneff->SetXTitle("GCT #eta");
  rctNisoEmIneff->SetYTitle("GCT #phi");
  rctNisoEmIneff->SetMinimum(0.005);
  rctNisoEmIneff->SetMaximum(1.0);
  rctNisoEmIneff->Draw("colz");
  dummybox->Draw("box, same");
  c1->SaveAs("./run" + runNumber + "/nonIsoInEff.png");

  gStyle->SetPalette (paletteSize, pOvereff);

  rctNisoEmOvereff->SetTitle("Nonisolated electron overefficiency");
  rctNisoEmOvereff->SetXTitle("GCT #eta");
  rctNisoEmOvereff->SetYTitle("GCT #phi");
  rctNisoEmOvereff->SetMinimum(0.005);
  rctNisoEmOvereff->SetMaximum(1.0);
  rctNisoEmOvereff->Draw("colz");
  dummybox->Draw("box, same");
  c1->SaveAs("./run" + runNumber + "/nonIsoOverEff.png");

  f->cd("run" + runNumber + ".root:/DQMData/L1TEMU/L1TdeRCT/RegionData/ServiceData");

  gStyle->SetOptStat("e");

  rctRegDataOcc2D->SetTitle("Region occupancy from data");
  rctRegDataOcc2D->SetXTitle("GCT #eta");
  rctRegDataOcc2D->SetYTitle("GCT #phi");
  rctRegDataOcc2D->Draw("box");
  c1->SaveAs("./run" + runNumber + "/regOccData.png");

  rctRegEmulOcc2D->SetTitle("Region occupancy from emulator");
  rctRegEmulOcc2D->SetXTitle("GCT #eta");
  rctRegEmulOcc2D->SetYTitle("GCT #phi");
  rctRegEmulOcc2D->Draw("box");
  c1->SaveAs("./run" + runNumber + "/regOccEmul.png");

  f->cd("run" + runNumber + ".root:/DQMData/L1TEMU/L1TdeRCT/RegionData");

  gStyle->SetOptStat (0);
  gStyle->SetPalette (paletteSize, pEff);

  rctRegEff2D->SetTitle("Region efficiency 1");
  rctRegEff2D->SetXTitle("GCT #eta");
  rctRegEff2D->SetYTitle("GCT #phi");
  rctRegEff2D->SetMinimum(0.005);
  rctRegEff2D->SetMaximum(1.0);
  rctRegEff2D->Draw("colz");
  dummybox->Draw("box, same");
  c1->SaveAs("./run" + runNumber + "/regEff1.png");

  rctRegSpEff2D->SetTitle("Region efficiency 2");
  rctRegSpEff2D->SetXTitle("GCT #eta");
  rctRegSpEff2D->SetYTitle("GCT #phi");
  rctRegSpEff2D->SetMinimum(0.005);
  rctRegSpEff2D->SetMaximum(1.0);
  rctRegSpEff2D->Draw("colz");
  dummybox->Draw("box, same");
  c1->SaveAs("./run" + runNumber + "/regEff2.png");

  gStyle->SetPalette (paletteSize, pIneff);

  rctRegIneff2D->SetTitle("Region inefficiency");
  rctRegIneff2D->SetXTitle("GCT #eta");
  rctRegIneff2D->SetYTitle("GCT #phi");
  rctRegIneff2D->SetMinimum(0.005);
  rctRegIneff2D->SetMaximum(1.0);
  rctRegIneff2D->Draw("colz");
  dummybox->Draw("box, same");
  c1->SaveAs("./run" + runNumber + "/regInEff.png");

  gStyle->SetPalette (paletteSize, pOvereff);

  rctRegOvereff2D->SetTitle("Region overefficiency");
  rctRegOvereff2D->SetXTitle("GCT #eta");
  rctRegOvereff2D->SetYTitle("GCT #phi");
  rctRegOvereff2D->SetMinimum(0.005);
  rctRegOvereff2D->SetMaximum(1.0);
  rctRegOvereff2D->Draw("colz");
  dummybox->Draw("box, same");
  c1->SaveAs("./run" + runNumber + "/regOverEff.png");
}
EOF
fi

if [ ! -f $CMSSW_BASE/src/DQM/L1TMonitor/test/effCurve.C ]
then
  cat << EOF > "$CMSSW_BASE/src/DQM/L1TMonitor/test/effCurve.C"
void effcurve()
{


TH1F *sum1 = new TH1F("sum1","sum1",64,-0.5,63.5) ;
TH1F *sum2 = new TH1F("sum2","sum2",64,-0.5,63.5) ;
TH1F *result = new TH1F("NisoEm Efficiency","NisoEm Efficiency",64,-0.5,63.5) ;

 //sum1->Sumw2() ;
// sum2->Sumw2() ;
result->Sumw2() ;

for(int i=0;i<16;i++)
{
 for(int j=0; j<18; j++)
 {


char name1[160];
char name2[160];
char channel[80]={""};
bool runit = true ;

strcpy(name1,"run69240.MUONS.root:/DQMData/L1TEMU/L1TdeRCT/EffCurves/NisoEm/ServiceData/SingleChannels/EemulChnlTrig") ;
strcpy(name2,"run69240.MUONS.root:/DQMData/L1TEMU/L1TdeRCT/EffCurves/NisoEm/ServiceData/SingleChannels/EemulChnl") ;

     if(i<10 && j<10) sprintf(channel,"_0%d0%d",i,j);
     else if(i<10) sprintf(channel,"_0%d%d",i,j);
      else if(j<10) sprintf(channel,"_%d0%d",i,j);
       else sprintf(channel,"_%d%d",i,j);


     strcat(name1,channel);
     strcat(name2,channel);

//std::cout << " name1 " << name1 << std::endl ;
//std::cout << " name2 " << name2 << std::endl ;

TH1F *tmp1 = (TH1F*)f.Get(name1) ;
TH1F *tmp2 = (TH1F*)f.Get(name2) ;

// for EEG runs
 if(j==17 && i==13 ) runit = false ;
 if(j==15 && (i==11 || i==12 || i==13 || i==14) ) runit = false ;
 if(j==13 && i==8 ) runit = false ;
 if(j==12 && i==10 ) runit = false ;
 if(j==11 && i==9 ) runit = false ;
 if(j==9 && (i==7 || i==8) ) runit = false ;
 if(j==8 && i==8 ) runit = false ;
 if(j==7 && (i==11 || i==12 || i==13 || i==14) ) runit = false ;
 if(j==6 && (i==7 || i==11 || i==12 || i==13 || i==14) ) runit = false ;
 if(j==5 && (i==7 || i==8) ) runit = false ;
 if(j==4 && i==14 ) runit = false ;
 if(j==3 && i==10 ) runit = false ;
 if(j==1 && i==7 ) runit = false ;
 if(j==0 && i==14 ) runit = false ;

//for HEG runs
//if( (j==11 || j==12) && (i>=4 && i<=10) ) runit = false ;
//if(i==14 && j==2 ) runit = false ;
//if(i==15 && j==3 ) runit = false ;
//if(i==7 && j==10 ) runit = false ;
//if(i==10 && j==14 ) runit = false ;
//if(i==7 && j==1 ) runit = false ;
//if(i==12 && j==9 ) runit = false ;
//if(i==14 && j==16 ) runit = false ;

//if(i==9 && j==13 ) runit = false ;

if(runit)
{
sum1->Add(tmp1) ;
sum2->Add(tmp2) ;
}


 }
}


   const int nbinsx = sum1 -> GetNbinsX();

   for(int binx = 1; binx <= nbinsx; ++binx)
   {
      const double ratio = sum1 -> GetBinContent(binx);
      const double staterr = sum1 -> GetBinError(binx);

//      if(ratio > 0.0)
//      {
//  const double err = ratio * sqrt(fracerr*fracerr + (staterr/ratio)*(staterr/ratio));
  const double err = 0.0 ;
         sum1 -> SetBinError(binx, err);
//      }
   }


result->Divide(sum2,sum1,1.,1.,"E") ;

gStyle->SetOptStat(0); 

result->SetXTitle("Emulated RCT NisoEm Rank ");
result->SetFillColor(42) ;
result->Draw("hist") ;

result->SetMarkerStyle(20) ;
result->Draw("epsame") ;
}
EOF
fi

bsub -q 8nh < lsfJob.csh
