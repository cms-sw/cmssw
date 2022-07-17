#include "TROOT.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TTree.h"
#include "TChain.h"

#include <iostream>
#include <algorithm>
#include <vector>
#include <map>
#include <iostream>
#include <iomanip>  // std::setw

//**********************************************//
// Auxilliary class
//**********************************************//
class RecordInfo : public TNamed {
public:
  RecordInfo(const char* record, const char* tag) : TNamed(record, tag) {}

  void printInfo() const { std::cout << GetName() << " " << GetTitle() << std::endl; }

  const char* getRecord() { return this->GetName(); }

  const char* getIOVSince() { return this->GetTitle(); }
};

enum ModuleGeometry {
  UNKNOWNGEOMETRY,
  IB1,
  IB2,
  OB1,
  OB2,
  W1A,
  W2A,
  W3A,
  W1B,
  W2B,
  W3B,
  W4,
  W5,
  W6,
  W7,
  END_OF_GEOMETRIES
};

enum TrackerRegion {
  TIB1 = 1,
  TIB2 = 2,
  TIB3 = 3,
  TIB4 = 4,
  TOB1 = 5,
  TOB2 = 6,
  TOB3 = 7,
  TOB4 = 8,
  TOB5 = 9,
  TOB6 = 10,
  TIDP1 = 11,
  TIDP2 = 12,
  TIDP3 = 13,
  TIDM1 = 14,
  TIDM2 = 15,
  TIDM3 = 16,
  TECP1 = 17,
  TECP2 = 18,
  TECP3 = 19,
  TECP4 = 20,
  TECP5 = 21,
  TECP6 = 22,
  TECP7 = 23,
  TECP8 = 24,
  TECP9 = 25,
  TECM1 = 26,
  TECM2 = 27,
  TECM3 = 28,
  TECM4 = 29,
  TECM5 = 30,
  TECM6 = 31,
  TECM7 = 32,
  TECM8 = 33,
  TECM9 = 34,
  END_OF_REGIONS = 35
};

/*--------------------------------------------------------------------*/
TrackerRegion getTheRegionFromTopology(int subdet, int side, int layer)
/*--------------------------------------------------------------------*/
{
  int ret(-99);
  switch (subdet) {
    case 3:
      // this is TIB
      ret = layer;
      break;
    case 4:
      // this is TID
      ret = side == 1 ? 10 + std::abs(layer) : 13 + std::abs(layer);
      break;
    case 5:
      // this is TOB
      ret = layer + 4;
      break;
    case 6:
      // this is TEC
      ret = side == 1 ? 16 + std::abs(layer) : 25 + std::abs(layer);
      break;
    default:
      std::cout << "getTheRegionFromTopology(): shall never ever be here!" << std::endl;
      break;
  }
  return static_cast<TrackerRegion>(ret);
}

/*--------------------------------------------------------------------*/
const char* regionType(int index)
/*--------------------------------------------------------------------*/
{
  auto region = static_cast<std::underlying_type_t<TrackerRegion> >(index);

  switch (region) {
    case TIB1:
      return "TIB L1";
    case TIB2:
      return "TIB L2";
    case TIB3:
      return "TIB L3";
    case TIB4:
      return "TIB L4";
    case TOB1:
      return "TOB L1";
    case TOB2:
      return "TOB L2";
    case TOB3:
      return "TOB L3";
    case TOB4:
      return "TOB L4";
    case TOB5:
      return "TOB L5";
    case TOB6:
      return "TOB L6";
    case TIDP1:
      return "TID+ D1";
    case TIDP2:
      return "TID+ D2";
    case TIDP3:
      return "TID+ D3";
    case TIDM1:
      return "TID- D1";
    case TIDM2:
      return "TID- D2";
    case TIDM3:
      return "TID- D3";
    case TECP1:
      return "TEC+ D1";
    case TECP2:
      return "TEC+ D2";
    case TECP3:
      return "TEC+ D3";
    case TECP4:
      return "TEC+ D4";
    case TECP5:
      return "TEC+ D5";
    case TECP6:
      return "TEC+ D6";
    case TECP7:
      return "TEC+ D7";
    case TECP8:
      return "TEC+ D8";
    case TECP9:
      return "TEC+ D9";
    case TECM1:
      return "TEC- D1";
    case TECM2:
      return "TEC- D2";
    case TECM3:
      return "TEC- D3";
    case TECM4:
      return "TEC- D4";
    case TECM5:
      return "TEC- D5";
    case TECM6:
      return "TEC- D6";
    case TECM7:
      return "TEC- D7";
    case TECM8:
      return "TEC- D8";
    case TECM9:
      return "TEC- D9";
    case END_OF_REGIONS:
      return "undefined";
    default:
      return "should never be here";
  }
}

/*--------------------------------------------------------------------*/
const char* moduleType(int index)
/*--------------------------------------------------------------------*/
{
  auto geometry = static_cast<std::underlying_type_t<ModuleGeometry> >(index);

  switch (geometry) {
    case UNKNOWNGEOMETRY:
      return "unknown geometry";
    case IB1:
      return "IB1";
    case IB2:
      return "IB2";
    case OB1:
      return "OB1";
    case OB2:
      return "OB2";
    case W1A:
      return "W1A";
    case W2A:
      return "W2A";
    case W3A:
      return "W3A";
    case W1B:
      return "W1B";
    case W2B:
      return "W2B";
    case W3B:
      return "W3B";
    case W4:
      return "W4";
    case W5:
      return "W5";
    case W6:
      return "W6";
    case W7:
      return "W7";
    case END_OF_GEOMETRIES:
      return "NONE";
    default:
      return "should never be here";
  }
}

/*--------------------------------------------------------------------*/
void readNSiStripDBTrees(TString fname)
/*--------------------------------------------------------------------*/
{
  TChain* tree_ = new TChain("treeDump/StripDBTree");
  tree_->Add(fname);

  uint32_t detId_, ring_, istrip_, det_type_;
  Int_t layer_, side_, subdetId_;
  float pedestal_, noise_, gsim_, g1_, g2_, lenght_;
  bool isTIB_, isTOB_, isTEC_, isTID_, isBad_;

  std::map<int, TH1F*> PedestalPerLayer;
  std::map<int, TH1F*> idealNoiseRatioPerLayer;
  std::map<int, TH1F*> NoisePerLayer;
  std::map<int, TH1F*> g1PerLayer;
  std::map<int, TH2F*> noiseVsG1PerModuleGeometry;
  std::map<int, TProfile*> p_noiseVsG1PerModuleGeometry;

  tree_->SetBranchAddress("detId", &detId_);
  tree_->SetBranchAddress("detType", &det_type_);
  tree_->SetBranchAddress("noise", &noise_);
  tree_->SetBranchAddress("pedestal", &pedestal_);
  tree_->SetBranchAddress("istrip", &istrip_);
  tree_->SetBranchAddress("gsim", &gsim_);
  tree_->SetBranchAddress("g1", &g1_);
  tree_->SetBranchAddress("g2", &g2_);
  tree_->SetBranchAddress("layer", &layer_);
  tree_->SetBranchAddress("side", &side_);
  tree_->SetBranchAddress("subdetId", &subdetId_);
  tree_->SetBranchAddress("ring", &ring_);
  tree_->SetBranchAddress("length", &lenght_);
  tree_->SetBranchAddress("isBad", &isBad_);
  tree_->SetBranchAddress("isTIB", &isTIB_);
  tree_->SetBranchAddress("isTOB", &isTOB_);
  tree_->SetBranchAddress("isTEC", &isTEC_);
  tree_->SetBranchAddress("isTID", &isTID_);

  int nentries = tree_->GetEntries();
  std::cout << "Number of entries = " << nentries << std::endl;

  tree_->LoadTree(0);
  TObjString* s = (TObjString*)tree_->GetTree()->GetUserInfo()->At(0);

  //RecordInfo *header = (RecordInfo*)tree_->GetTree()->GetUserInfo()->FindObject("SiStripPedestalsRcd");
  //header->printInfo();
  //std::cout << "printing recordInfo:"<<header->getRecord() << " IOV: "<< header->getIOVSince() << std::endl;

  // print the headers

  std::vector<const char*> records = {
      "SiStripPedestalsRcd", "SiStripNoisesRcd", "SiStripApvGainRcd", "SiStripApvGain2Rcd", "SiStripQualityRcd"};
  for (const auto& rec : records) {
    RecordInfo* header = (RecordInfo*)tree_->GetTree()->GetUserInfo()->FindObject(rec);
    //header->printInfo();
    std::cout << "printing recordInfo: " << header->getRecord() << " IOV: " << header->getIOVSince() << std::endl;
  }

  TH1F* h_avgPedestal =
      new TH1F("h_avgPedestal_perRegion", "average Pedestal per region;;average Pedestals [ADC counts]", 34, 0., 34.);
  TH1F* h_avgIdealNoiseRatio =
      new TH1F("h_avgIdealNoise_perRegion", "average Ideal Noise per region;;averag Ideal Noise ratio", 34, 0., 34.);
  TH1F* h_avgNoise =
      new TH1F("h_avgNoise_perRegion", "average Noise per region;; average Noise [ADC counts]", 34, 0., 34.);

  TH1F* h_Pedestal = new TH1F("h_Pedestal", "Pedestal;Pedestals [ADC counts];n. strips", 300, 0., 300.);
  TH1F* h_idealNoiseRatio = new TH1F("h_IdealNoise", "Ideal Noise;Ideal Noise ratio;n. strips", 500, 0., 10.);
  TH1F* h_Noise = new TH1F("h_Noise", "Noise;Noise [ADC counts];n. strips", 500, 0., 10.);

  TH2F* h2_NoiseVsPedestal = new TH2F(
      "h2_NoiseVsPedestal", "Noise Vs Pedestal;Pedestals [ADC counts];Noise [ACD counts]", 350, 0., 350., 120, 0., 12.);
  TH2F* h2_NoiseVsG1 =
      new TH2F("h2_NoiseVsG1", "Noise vs G1 Gain;G1 gain;Noise [ACD counts]", 100, 0., 2., 200, 0., 20.);

  TH2F* h2_NoiseVsPedestalTIB = new TH2F("h2_NoiseVsPedestalTIB",
                                         "Noise Vs Pedestal;Pedestals [ADC counts];Noise [ACD counts]",
                                         350,
                                         0.,
                                         350.,
                                         120,
                                         0.,
                                         12.);
  TH2F* h2_NoiseVsPedestalTOB = new TH2F("h2_NoiseVsPedestalTOB",
                                         "Noise Vs Pedestal;Pedestals [ADC counts];Noise [ACD counts]",
                                         350,
                                         0.,
                                         350.,
                                         120,
                                         0.,
                                         12.);
  TH2F* h2_NoiseVsPedestalTID = new TH2F("h2_NoiseVsPedestalTID",
                                         "Noise Vs Pedestal;Pedestals [ADC counts];Noise [ACD counts]",
                                         350,
                                         0.,
                                         350.,
                                         120,
                                         0.,
                                         12.);
  TH2F* h2_NoiseVsPedestalTEC = new TH2F("h2_NoiseVsPedestalTEC",
                                         "Noise Vs Pedestal;Pedestals [ADC counts];Noise [ACD counts]",
                                         350,
                                         0.,
                                         350.,
                                         120,
                                         0.,
                                         12.);

  TH1F* h_g1 = new TH1F("h_g1", "g1 gain;g1 gain;n. strips", 200, 0., 2.);

  // loop on the tracker regions
  for (int region = TrackerRegion::TIB1; region != TrackerRegion::END_OF_REGIONS; region++) {
    auto tag = regionType(region);
    std::cout << "booking region: " << std::setw(3) << region << " -> " << tag << std::endl;
    idealNoiseRatioPerLayer[region] = new TH1F(
        Form("IdealNoise_%s", tag), Form("Ideal Noise %s;Ideal Noise ratio for %s;n. strips", tag, tag), 500, 0., 10.);
    NoisePerLayer[region] =
        new TH1F(Form("Noise_%s", tag), Form("Noise %s;Noise for %s [ADC counts];n. strips", tag, tag), 500, 0., 10.);
    g1PerLayer[region] = new TH1F(Form("g1_%s", tag), Form("g1 %s;g1 for %s;n. strips", tag, tag), 200, 0., 2.);
    PedestalPerLayer[region] =
        new TH1F(Form("pedestal_%s", tag), Form("pedestal %s;pedestal for %s;n. strips", tag, tag), 350, 0., 350.);

    h_avgPedestal->GetXaxis()->SetBinLabel(region, tag);
    h_avgIdealNoiseRatio->GetXaxis()->SetBinLabel(region, tag);
    h_avgNoise->GetXaxis()->SetBinLabel(region, tag);
  }

  // loop on the tracker module geometries
  for (int geometry = ModuleGeometry::IB1; geometry != ModuleGeometry::END_OF_GEOMETRIES; geometry++) {
    auto tag = moduleType(geometry);
    noiseVsG1PerModuleGeometry[geometry] = new TH2F(Form("h2_NoiseVsG1_%s", tag),
                                                    Form("Noise vs G1 Gain for %s;G1 gain;Noise [ACD counts]", tag),
                                                    100,
                                                    0.,
                                                    2.,
                                                    200,
                                                    0.,
                                                    20.);
    p_noiseVsG1PerModuleGeometry[geometry] = new TProfile(
        Form("p_NoiseVsG1_%s", tag), Form("Noise vs G1 Gain for %s;G1 gain;Noise [ACD counts]", tag), 100, 0., 2.);
  }

  uint32_t cachedDetId = -1;

  printf("Progressing Bar              :0%%       20%%       40%%       60%%       80%%       100%%\n");
  printf("Scanning ntuple              :");
  int TreeStep = tree_->GetEntries() / 50;
  if (TreeStep == 0)
    TreeStep = 1;
  for (Int_t stripNo = 0; stripNo < nentries; stripNo++) {
    if (stripNo % TreeStep == 0) {
      printf(".");
      fflush(stdout);
    }
    Int_t IgetStrip = tree_->GetEntry(stripNo);
    auto region = getTheRegionFromTopology(subdetId_, side_, layer_);

    h2_NoiseVsPedestal->Fill(pedestal_, noise_);
    h2_NoiseVsG1->Fill(g1_, noise_);

    switch (subdetId_) {
      case 3:
        h2_NoiseVsPedestalTIB->Fill(pedestal_, noise_);
        break;
      case 5:
        h2_NoiseVsPedestalTOB->Fill(pedestal_, noise_);
        break;
      case 4:
        h2_NoiseVsPedestalTID->Fill(pedestal_, noise_);
        break;
      case 6:
        h2_NoiseVsPedestalTEC->Fill(pedestal_, noise_);
        break;
      default:
        std::cout << "shall never be here!" << std::endl;
    }

    h_Pedestal->Fill(pedestal_);
    h_idealNoiseRatio->Fill(noise_ / g1_);
    h_Noise->Fill(noise_);
    h_g1->Fill(g1_);

    PedestalPerLayer[region]->Fill(pedestal_);
    idealNoiseRatioPerLayer[region]->Fill(noise_ / g1_);
    NoisePerLayer[region]->Fill(noise_);
    g1PerLayer[region]->Fill(g1_);

    noiseVsG1PerModuleGeometry[det_type_]->Fill(g1_, noise_);
    p_noiseVsG1PerModuleGeometry[det_type_]->Fill(g1_, noise_);

    //std::cout << " strip n."<< stripNo << " detId:"<< detId_ << " strip n.: "<< istrip_ << std::endl;
    if (detId_ != cachedDetId) {
      //  std::cout << " strip n."<< stripNo << " detId:"<< detId_ << " strip n.: "<< istrip_
      //	<< " subdet: " << subdetId_ <<" side: "<< side_ << " layer: "<< layer_ << " (region: " << region << ") =>  " << regionType(region) << " " << moduleType(det_type_) << std::endl;
      cachedDetId = detId_;
    }
  }
  printf("\n");

  for (int region = TrackerRegion::TIB1; region != TrackerRegion::END_OF_REGIONS; region++) {
    h_avgIdealNoiseRatio->SetBinContent(region, idealNoiseRatioPerLayer[region]->GetMean());
    h_avgNoise->SetBinContent(region, NoisePerLayer[region]->GetMean());
    h_avgPedestal->SetBinContent(region, PedestalPerLayer[region]->GetMean());
  }

  TFile* outfile = TFile::Open(Form("idealNoise_%s.root", (s->GetString()).Data()), "RECREATE");
  outfile->cd();
  h_Pedestal->Write();
  h_idealNoiseRatio->Write();
  h_Noise->Write();
  h2_NoiseVsG1->Write();
  h2_NoiseVsPedestal->Write();
  h_g1->Write();

  h_avgPedestal->Write();
  h_avgIdealNoiseRatio->Write();
  h_avgNoise->Write();

  TDirectory* byPartition = outfile->mkdir("ByPartition");
  byPartition->cd();
  h2_NoiseVsPedestalTIB->Write();
  h2_NoiseVsPedestalTOB->Write();
  h2_NoiseVsPedestalTID->Write();
  h2_NoiseVsPedestalTEC->Write();

  TDirectory* cdIdealNoise = outfile->mkdir("idealNoise");
  cdIdealNoise->cd();
  for (int region = TrackerRegion::TIB1; region != TrackerRegion::END_OF_REGIONS; region++) {
    auto tag = regionType(region);
    idealNoiseRatioPerLayer[region]->Write();
  }

  outfile->cd();
  TDirectory* cdNoise = outfile->mkdir("Noise");
  cdNoise->cd();
  for (int region = TrackerRegion::TIB1; region != TrackerRegion::END_OF_REGIONS; region++) {
    auto tag = regionType(region);
    NoisePerLayer[region]->Write();
  }

  outfile->cd();
  TDirectory* cdG1 = outfile->mkdir("g1");
  cdG1->cd();
  for (int region = TrackerRegion::TIB1; region != TrackerRegion::END_OF_REGIONS; region++) {
    auto tag = regionType(region);
    g1PerLayer[region]->Write();
  }

  outfile->cd();
  TDirectory* cNoiseVsG1 = outfile->mkdir("tickmark_vs_noise");
  cNoiseVsG1->cd();
  for (int geometry = ModuleGeometry::IB1; geometry != ModuleGeometry::END_OF_GEOMETRIES; geometry++) {
    noiseVsG1PerModuleGeometry[geometry]->Write();
    p_noiseVsG1PerModuleGeometry[geometry]->Write();
  }

  outfile->Close();
  tree_->Delete();
  return;
}
