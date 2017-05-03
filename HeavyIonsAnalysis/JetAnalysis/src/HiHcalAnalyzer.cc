//---------------------------------------------------------------------------
#include <memory>
//---------------------------------------------------------------------------
#include "TTree.h"
//---------------------------------------------------------------------------
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
//---------------------------------------------------------------------------
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
//---------------------------------------------------------------------------
#include "DataFormats/METReco/interface/HcalNoiseSummary.h"
#include "DataFormats/METReco/interface/HcalNoiseRBX.h"
#include "RecoMET/METAlgorithms/interface/HcalHPDRBXMap.h"
//---------------------------------------------------------------------------
#include "FWCore/ParameterSet/interface/ParameterSet.h"
//---------------------------------------------------------------------------
class HiHcalAnalyzer : public edm::EDAnalyzer
{
public:
  explicit HiHcalAnalyzer(const edm::ParameterSet&);
  ~HiHcalAnalyzer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void beginJob();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  virtual void beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void endRun(edm::Run const&, edm::EventSetup const&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

private:
  edm::Service<TFileService> FileService;

private:
  void CleanUp();

private:
  edm::EDGetTokenT<HcalNoiseSummary> NoiseSummaryTag;
  //edm::InputTag NoiseRBXTag;

private:
  TTree *OutputTree;

  int RunNumber;
  long long EventNumber;
  int LumiSection;
  int Bunch;
  int Orbit;
  long long Time;

  int FilterStatus;
  int MaxZeros, MaxHPDHits, MaxHPDNoOtherHits, MaxRBXHits;
  int IsolatedCount, FlatNoiseCount, SpikeNoiseCount, TriangleNoiseCount;
  double IsolatedSumE, FlatNoiseSumE, SpikeNoiseSumE, TriangleNoiseSumE;
  double IsolatedSumET, FlatNoiseSumET, SpikeNoiseSumET, TriangleNoiseSumET;
  bool HasBadTS4TS5;
  double TotalCalibCharge;
  double MinE2E10, MaxE2E10;
  
  // double RBXEnergy[72], RBXEnergy15[72];
  // int RBXHitCount[72];
  // double RBXR45[72];
  // double RBXCharge[72][10];
};
//---------------------------------------------------------------------------
HiHcalAnalyzer::HiHcalAnalyzer(const edm::ParameterSet& iConfig)
{
  NoiseSummaryTag = consumes<HcalNoiseSummary>(iConfig.getUntrackedParameter<edm::InputTag>("NoiseSummaryTag"));
  // NoiseRBXTag = iConfig.getUntrackedParameter<edm::InputTag>("NoiseRBXTag");
}
//---------------------------------------------------------------------------
HiHcalAnalyzer::~HiHcalAnalyzer()
{
}
//---------------------------------------------------------------------------
void HiHcalAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  using namespace reco;

  // Clean-up variables
  CleanUp();

  // Basic event coordinates
  RunNumber = iEvent.id().run();
  EventNumber = iEvent.id().event();
  LumiSection = iEvent.luminosityBlock();
  Bunch = iEvent.bunchCrossing();
  Orbit = iEvent.orbitNumber();
  Time = iEvent.time().value();

  // Get stuff
  Handle<HcalNoiseSummary> hSummary;
  iEvent.getByToken(NoiseSummaryTag, hSummary);

  // Handle<HcalNoiseRBXCollection> hRBX;
  // iEvent.getByLabel(NoiseRBXTag, hRBX);

  // Check if the stuff we get is good.  If not...
  if(hSummary.isValid() == false
     //|| hRBX.isValid() == false
     )
  {
    // ...then we barf at user about bad file, but we still fill in a filler entry in the tree
    edm::LogError("DataNotFound") << "Hcal noise summary information is invalid for "
      "run " << RunNumber << " event " << EventNumber << " LS " << LumiSection;
    OutputTree->Fill();
    return;
  }

  // Dump information out of the summary object
  FilterStatus = hSummary->noiseFilterStatus();
  MaxZeros = hSummary->maxZeros();
  MaxHPDHits = hSummary->maxHPDHits();
  MaxHPDNoOtherHits = hSummary->maxHPDNoOtherHits();
  MaxRBXHits = hSummary->maxRBXHits();
  IsolatedCount = hSummary->numIsolatedNoiseChannels();
  IsolatedSumE = hSummary->isolatedNoiseSumE();
  IsolatedSumET = hSummary->isolatedNoiseSumEt();
  FlatNoiseCount = hSummary->numFlatNoiseChannels();
  FlatNoiseSumE = hSummary->flatNoiseSumE();
  FlatNoiseSumET = hSummary->flatNoiseSumEt();
  SpikeNoiseCount = hSummary->numSpikeNoiseChannels();
  SpikeNoiseSumE = hSummary->spikeNoiseSumE();
  SpikeNoiseSumET = hSummary->spikeNoiseSumEt();
  TriangleNoiseCount = hSummary->numTriangleNoiseChannels();
  TriangleNoiseSumE = hSummary->triangleNoiseSumE();
  TriangleNoiseSumET = hSummary->triangleNoiseSumEt();
  HasBadTS4TS5 = hSummary->HasBadRBXTS4TS5();
  TotalCalibCharge = hSummary->GetTotalCalibCharge();
  MinE2E10 = hSummary->minE2Over10TS();
  MaxE2E10 = hSummary->maxE2Over10TS();

  // // Dump information out of the RBX array
  // for(int iRBX = 0; iRBX < (int)hRBX->size(); iRBX++)
  // {
  //   int ID = (*hRBX)[iRBX].idnumber();
  //   if(ID >= 72)   // WTF!
  //     continue;

  //   RBXEnergy[ID] = (*hRBX)[iRBX].recHitEnergy();
  //   RBXEnergy15[ID] = (*hRBX)[iRBX].recHitEnergy(1.5);
  //   RBXHitCount[ID] = (*hRBX)[iRBX].numRecHits(1.5);

  //   std::vector<float> allcharge = (*hRBX)[iRBX].allCharge();
  //   for(int iTS = 0; iTS < 10 && iTS < (int)allcharge.size(); iTS++)
  //     RBXCharge[ID][iTS] = allcharge[iTS];

  //   if(RBXCharge[ID][4] + RBXCharge[ID][5] > 1)
  //     RBXR45[ID] = (RBXCharge[ID][4] - RBXCharge[ID][5]) / (RBXCharge[ID][4] + RBXCharge[ID][5]);
  //   else
  //     RBXR45[ID] = -9999;
  // }

  // Finally fill the tree
  OutputTree->Fill();
}
//---------------------------------------------------------------------------
void HiHcalAnalyzer::beginJob()
{
  OutputTree = FileService->make<TTree>("hbhenoise", "v1");

  OutputTree->Branch("run", &RunNumber, "run/I");
  OutputTree->Branch("event", &EventNumber, "event/LL");
  OutputTree->Branch("luminosityBlock", &LumiSection, "luminosityBlock/I");
  OutputTree->Branch("bunchCrossing", &Bunch, "bunchCrossing/I");
  OutputTree->Branch("orbit", &Orbit, "orbit/I");
  OutputTree->Branch("time", &Time, "time/LL");


  OutputTree->Branch("FilterStatus", &FilterStatus, "FilterStatus/I");
  OutputTree->Branch("MaxZeros", &MaxZeros, "MaxZeros/I");
  OutputTree->Branch("MaxHPDHits", &MaxHPDHits, "MaxHPDHits/I");
  OutputTree->Branch("MaxHPDNoOtherHits", &MaxHPDNoOtherHits, "MaxHPDNoOtherHits/I");
  OutputTree->Branch("MaxRBXHits", &MaxRBXHits, "MaxRBXHits/I");
  OutputTree->Branch("IsolatedCount", &IsolatedCount, "IsolatedCount/I");
  OutputTree->Branch("IsolatedSumE", &IsolatedSumE, "IsolatedSumE/D");
  OutputTree->Branch("IsolatedSumET", &IsolatedSumET, "IsolatedSumET/D");
  OutputTree->Branch("FlatNoiseCount", &FlatNoiseCount, "FlatNoiseCount/I");
  OutputTree->Branch("FlatNoiseSumE", &FlatNoiseSumE, "FlatNoiseSumE/D");
  OutputTree->Branch("FlatNoiseSumET", &FlatNoiseSumET, "FlatNoiseSumET/D");
  OutputTree->Branch("SpikeNoiseCount", &SpikeNoiseCount, "SpikeNoiseCount/I");
  OutputTree->Branch("SpikeNoiseSumE", &SpikeNoiseSumE, "SpikeNoiseSumE/D");
  OutputTree->Branch("SpikeNoiseSumET", &SpikeNoiseSumET, "SpikeNoiseSumET/D");
  OutputTree->Branch("TriangleNoiseCount", &TriangleNoiseCount, "TriangleNoiseCount/I");
  OutputTree->Branch("TriangleNoiseSumE", &TriangleNoiseSumE, "TriangleNoiseSumE/D");
  OutputTree->Branch("TriangleNoiseSumET", &TriangleNoiseSumET, "TriangleNoiseSumET/D");
  OutputTree->Branch("HasBadTS4TS5", &HasBadTS4TS5, "HasBadTS4TS5/O");
  OutputTree->Branch("TotalCalibCharge", &TotalCalibCharge, "TotalCalibCharge/D");
  OutputTree->Branch("MinE2E10", &MinE2E10, "MinE2E10/D");
  OutputTree->Branch("MaxE2E10", &MaxE2E10, "MaxE2E10/D");

  // OutputTree->Branch("RBXEnergy", RBXEnergy, "RBXEnergy[72]/D");
  // OutputTree->Branch("RBXEnergy15", RBXEnergy15, "RBXEnergy15[72]/D");
  // OutputTree->Branch("RBXHitCount", RBXHitCount, "RBXHitCount[72]/I");
  // OutputTree->Branch("RBXR45", RBXR45, "RBXR45[72]/D");
  // OutputTree->Branch("RBXCharge", RBXCharge, "RBXCharge[72][10]/D");

}
//---------------------------------------------------------------------------
void HiHcalAnalyzer::endJob()
{
}
//---------------------------------------------------------------------------
void HiHcalAnalyzer::beginRun(edm::Run const&, edm::EventSetup const&)
{
}
//---------------------------------------------------------------------------
void HiHcalAnalyzer::endRun(edm::Run const&, edm::EventSetup const&)
{
}
//---------------------------------------------------------------------------
void HiHcalAnalyzer::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
//---------------------------------------------------------------------------
void HiHcalAnalyzer::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
//---------------------------------------------------------------------------
void HiHcalAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
  // The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}
//---------------------------------------------------------------------------
void HiHcalAnalyzer::CleanUp()
{
  RunNumber = -1;
  EventNumber = -1;
  LumiSection = -1;
  Bunch = -1;
  Orbit = -1;
  Time = -1;

  FilterStatus = -1;
  MaxZeros = -1;
  MaxHPDHits = -1;
  MaxHPDNoOtherHits = -1;
  MaxRBXHits = -1;
  IsolatedCount = -1;
  FlatNoiseCount = -1;
  IsolatedSumE = -1;
  FlatNoiseSumE = -1;
  IsolatedSumET = -1;
  FlatNoiseSumET = -1;
  SpikeNoiseCount = -1;
  SpikeNoiseSumE = -1;
  SpikeNoiseSumET = -1;
  TriangleNoiseCount = -1;
  TriangleNoiseSumE = -1;
  TriangleNoiseSumET = -1;
  HasBadTS4TS5 = false;
  TotalCalibCharge = -1;
  MinE2E10 = -1;
  MaxE2E10 = -1;

  // for(int iID = 0; iID < 72; iID++)
  // {
  //   RBXEnergy[iID] = -1;
  //   RBXEnergy15[iID] = -1;
  //   RBXHitCount[iID] = -1;
  //   RBXR45[iID] = 9999;
  //   for(int iTS = 0; iTS < 10; iTS++)
  //     RBXCharge[iID][iTS] = -1;
  // }
}
//---------------------------------------------------------------------------
//define this as a plug-in
DEFINE_FWK_MODULE(HiHcalAnalyzer);
