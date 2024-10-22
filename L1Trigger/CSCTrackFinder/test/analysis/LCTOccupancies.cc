
// system include files
#include <memory>
#include <cmath>
#include <iostream>
#include <sstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"  //
#include "DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h"
#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCTrackFinderDataTypes.h"

#include "L1Trigger/CSCTrackFinder/test/src/RunSRLUTs.h"

#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "DataFormats/Luminosity/interface/LumiSummary.h"

#include "TH1F.h"
#include "TFile.h"
#include "TTree.h"

using namespace std;
using namespace edm;

class LCTOccupancies : public edm::one::EDAnalyzer<edm::one::SharedResources, edm::one::WatchLuminosityBlocks> {
public:
  explicit LCTOccupancies(const edm::ParameterSet&);
  ~LCTOccupancies() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
  void beginLuminosityBlock(edm::LuminosityBlock const& iLumiBlock, edm::EventSetup const& iSetup) override;

  // ----------member data ---------------------------

  //TH1F* rateHist;

  edm::InputTag lctsTag;
  edm::InputTag vertexColTag;
  edm::Service<TFileService> fs;
  csctf_analysis::RunSRLUTs* runSRLUTs;
  float insLumi;
  int nVertex;
  bool haveRECO;
  int singleSectorNum;
  std::string outTreeFileName;

  TH1F* hNVertex;
  TH1F* hInsLumi;

  TH1F* hMPCLink;
  TH1F* hLocalPhi;
  TH1F* hPhi;
  TH1F* hEta;
  TH1F* hPhiPacked;
  TH1F* hEtaPacked;
  TH1F* hBx;

  TH1F* hSector;
  TH1F* hStation;
  TH1F* hEndcap;
  TH1F* hSubSector;

  TH1F* hOccStation1SubSec1;
  TH1F* hOccStation1SubSec2;
  TH1F* hOccStation2;
  TH1F* hOccStation3;
  TH1F* hOccStation4;
  TH1F* hOccMax;
  TH1F* hOccMaxNo0;
  TH1F* hStubsTotal;

  TH1F* hOccME11a;
  TH1F* hOccME11b;
  TH1F* hOccME12;
  TH1F* hOccME13;
  TH1F* hOccME21;
  TH1F* hOccME22;
  TH1F* hOccME31;
  TH1F* hOccME32;
  TH1F* hOccME41;
  TH1F* hOccME42;
  TH1F* hOccME42SingleSector;

  std::vector<TH1F*> hOccME1ChambsSubSec1;
  std::vector<TH1F*> hOccME1ChambsSubSec2;
  std::vector<TH1F*> hOccME2Chambs;
  std::vector<TH1F*> hOccME3Chambs;
  std::vector<TH1F*> hOccME4Chambs;

  //Tree stuff
  TFile* treeFile;
  TTree* tree;
  std::vector<int> occStation1SubSec1;
  std::vector<int> occStation1SubSec2;
  std::vector<int> occStation2;
  std::vector<int> occStation3;
  std::vector<int> occStation4;

  std::vector<std::vector<int> > occME1ChamberSubSec1;
  std::vector<std::vector<int> > occME1ChamberSubSec2;
  std::vector<std::vector<int> > occME2Chamber;
  std::vector<std::vector<int> > occME3Chamber;
  std::vector<std::vector<int> > occME4Chamber;
};
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
LCTOccupancies::LCTOccupancies(const edm::ParameterSet& iConfig)

{
  usesResource(TFileService::kSharedResource);

  //now do what ever initialization is needed
  runSRLUTs = new csctf_analysis::RunSRLUTs();
  lctsTag = iConfig.getParameter<edm::InputTag>("lctsTag");
  vertexColTag = iConfig.getParameter<edm::InputTag>("vertexColTag");
  outTreeFileName = iConfig.getUntrackedParameter<std::string>("outTreeFileName");
  haveRECO = iConfig.getUntrackedParameter<bool>("haveRECO");
  singleSectorNum = iConfig.getUntrackedParameter<int>("singleSectorNum");

  treeFile = new TFile(outTreeFileName.c_str(), "RECREATE");
  tree = new TTree("tree", "tree");

  occStation1SubSec1 = std::vector<int>(12, 0);
  occStation1SubSec2 = std::vector<int>(12, 0);
  occStation2 = std::vector<int>(12, 0);
  occStation3 = std::vector<int>(12, 0);
  occStation4 = std::vector<int>(12, 0);

  for (int iSector = 0; iSector < 12; iSector++) {
    occME1ChamberSubSec1.push_back(std::vector<int>(12, 0));
    occME1ChamberSubSec2.push_back(std::vector<int>(12, 0));
    occME2Chamber.push_back(std::vector<int>(9, 0));
    occME3Chamber.push_back(std::vector<int>(9, 0));
    occME4Chamber.push_back(std::vector<int>(9, 0));
  }

  tree->Branch("nVertex", &nVertex, "nVertex/I");
  tree->Branch("insLumi", &insLumi, "insLumi/F");

  tree->Branch("occStation1SubSec1", "vector<int>", &occStation1SubSec1);
  tree->Branch("occStation1SubSec2", "vector<int>", &occStation1SubSec2);
  tree->Branch("occStation2", "vector<int>", &occStation2);
  tree->Branch("occStation3", "vector<int>", &occStation3);
  tree->Branch("occStation4", "vector<int>", &occStation4);

  tree->Branch("occME1ChamberSubSec1", "vector<std::vector<int> >", &occME1ChamberSubSec1);
  tree->Branch("occME1ChamberSubSec2", "vector<std::vector<int> >", &occME1ChamberSubSec2);
  tree->Branch("occME2Chamber", "vector<std::vector<int> >", &occME2Chamber);
  tree->Branch("occME3Chamber", "vector<std::vector<int> >", &occME3Chamber);
  tree->Branch("occME4Chamber", "vector<std::vector<int> >", &occME4Chamber);
}

LCTOccupancies::~LCTOccupancies() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  treeFile->cd();
  tree->Write();
  delete runSRLUTs;
}

//
// member functions
//

// ------------ method called to for each event  ------------
void LCTOccupancies::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  ///////////////////
  //Setup Stuff//////
  ///////////////////
  nVertex = 0;

  std::vector<csctf::TrackStub> stubs;
  std::vector<csctf::TrackStub>::const_iterator stub;

  //MuonDigiCollection<CSCDetId,CSCCorrelatedLCTDigi>    "csctfunpacker"          ""        "CsctfFilter"
  edm::Handle<CSCCorrelatedLCTDigiCollection> lctDigiColls;
  iEvent.getByLabel(lctsTag, lctDigiColls);

  std::vector<csctf::TrackStub>* trackStubs = new std::vector<csctf::TrackStub>;
  runSRLUTs->makeTrackStubs(lctDigiColls.product(), trackStubs);

  std::vector<csctf::TrackStub>::const_iterator ts = trackStubs->begin();
  std::vector<csctf::TrackStub>::const_iterator tsEnd = trackStubs->end();

  for (unsigned i = 0; i < 12; i++) {
  }

  std::vector<int> occME11a(24, 0);
  std::vector<int> occME11b(24, 0);
  std::vector<int> occME12(24, 0);
  std::vector<int> occME13(24, 0);
  std::vector<int> occME21(12, 0);
  std::vector<int> occME22(12, 0);
  std::vector<int> occME31(12, 0);
  std::vector<int> occME32(12, 0);
  std::vector<int> occME41(12, 0);
  std::vector<int> occME42(12, 0);
  int occME42SingleSector = 0;

  for (int iSectors = 0; iSectors < 12; iSectors++) {
    occStation1SubSec1[iSectors] = 0;
    occStation1SubSec2[iSectors] = 0;
    occStation2[iSectors] = 0;
    occStation3[iSectors] = 0;
    occStation4[iSectors] = 0;
    for (int iChambers = 0; iChambers < 12; iChambers++) {
      occME1ChamberSubSec1[iSectors][iChambers] = 0;
      occME1ChamberSubSec2[iSectors][iChambers] = 0;
      if (iChambers < 9) {
        occME2Chamber[iSectors][iChambers] = 0;
        occME3Chamber[iSectors][iChambers] = 0;
        occME4Chamber[iSectors][iChambers] = 0;
      }
    }
  }

  for (; ts != tsEnd; ts++) {
    //std::cout << "etaValue: \t" <<ts->etaValue()<< std::endl;
    //std::cout << "phiValue: \t" <<ts->phiValue()<< std::endl;
    //std::cout << "eta: \t" <<ts->etaPacked()<< std::endl;
    //std::cout << "phi: \t" <<ts->phiPacked()<< std::endl;
    //std::cout << "cscid: \t" <<ts->cscid()<< std::endl;
    //std::cout << "subsector: \t" <<ts->subsector()<< std::endl;
    //std::cout << "sector: \t" <<ts->sector()<< std::endl;
    //std::cout << "station: \t" <<ts->station()<< std::endl;
    //std::cout << "endcap: \t" <<ts->endcap()<< std::endl;
    //std::cout << "bx: \t" <<ts->BX()<< std::endl;
    //std::cout << "MPCLink: \t" <<ts->getMPCLink()<< std::endl;
    //std::cout << std::endl;

    unsigned sector = ts->sector() - 1;

    if (ts->BX() != 0)
      continue;
    hMPCLink->Fill(ts->getMPCLink());
    hLocalPhi->Fill(ts->phiValue());
    hPhi->Fill(ts->phiValue() + 15.0 * M_PI / 180 + (sector)*60.0 * M_PI / 180);
    hPhiPacked->Fill(ts->phiPacked());
    hEta->Fill(ts->etaValue());
    hEtaPacked->Fill(ts->etaPacked());
    hBx->Fill(ts->BX());
    int station = ts->station();
    int subsector = ts->subsector();
    //CSCDetId detId(ts->getDetId().rawId());
    CSCDetId detId(ts->getDetId());
    int ring = detId.ring();
    int triggerChamber = CSCTriggerNumbering::triggerCscIdFromLabels(detId);
    //ME11a ring==4 doesn't work, no ring==4 events ?
    //std::cout << "station: \t" <<ts->station()<< std::endl;
    //std::cout << "ring: \t" <<ring<< std::endl << std::endl;
    if (ts->endcap() == 2) {
      station = -station;
      sector = sector + 6;
    }
    hSector->Fill(sector + 1);
    hStation->Fill(station);
    hSubSector->Fill(subsector);

    //std::cout << "station: " << station << std::endl;
    //std::cout << "my sector packed: " << sector << std::endl;
    //std::cout << "trigger Chamber: " << triggerChamber << std::endl;

    station = abs(station);
    if (station == 1) {
      if (subsector == 1) {
        occME1ChamberSubSec1[sector][triggerChamber - 1]++;
        occStation1SubSec1[sector]++;
        if (ring == 4)
          occME11a[sector]++;
        else if (ring == 1)
          occME11b[sector]++;
        else if (ring == 2)
          occME12[sector]++;
        else if (ring == 3)
          occME13[sector]++;
      } else {
        occME1ChamberSubSec2[sector][triggerChamber - 1]++;
        occStation1SubSec2[sector]++;
        if (ring == 4)
          occME11a[sector + 12]++;
        else if (ring == 1)
          occME11b[sector + 12]++;
        else if (ring == 2)
          occME12[sector + 12]++;
        else if (ring == 3)
          occME13[sector + 12]++;
      }
    }
    if (station == 2) {
      occME2Chamber[sector][triggerChamber - 1]++;
      occStation2[sector]++;
      if (ring == 1)
        occME21[sector]++;
      else if (ring == 2)
        occME22[sector]++;
    }
    if (station == 3) {
      occME3Chamber[sector][triggerChamber - 1]++;
      occStation3[sector]++;
      if (ring == 1)
        occME31[sector]++;
      else if (ring == 2)
        occME32[sector]++;
    }
    if (station == 4) {
      occME4Chamber[sector][triggerChamber - 1]++;
      occStation4[sector]++;
      if (ring == 1)
        occME41[sector]++;
      else if (ring == 2) {
        occME42[sector]++;
        occME42SingleSector++;
      }
    }
  }

  int maxOcc = 0;

  for (unsigned iSector = 0; iSector < 12; iSector++) {
    if (singleSectorNum != -1 && singleSectorNum != (int)iSector)
      continue;
    hOccStation1SubSec1->Fill(occStation1SubSec1[iSector]);
    hOccStation1SubSec2->Fill(occStation1SubSec2[iSector]);
    hOccStation2->Fill(occStation2[iSector]);
    hOccStation3->Fill(occStation3[iSector]);
    hOccStation4->Fill(occStation4[iSector]);

    if (occStation1SubSec1[iSector] > maxOcc)
      maxOcc = occStation1SubSec1[iSector];
    if (occStation1SubSec2[iSector] > maxOcc)
      maxOcc = occStation1SubSec2[iSector];
    if (occStation2[iSector] > maxOcc)
      maxOcc = occStation2[iSector];
    if (occStation3[iSector] > maxOcc)
      maxOcc = occStation3[iSector];
    if (occStation4[iSector] > maxOcc)
      maxOcc = occStation4[iSector];
  }
  hOccMax->Fill(maxOcc);
  if (maxOcc > 0)
    hOccMaxNo0->Fill(maxOcc);

  hStubsTotal->Fill(trackStubs->size());

  for (unsigned iSector = 0; iSector < 24; iSector++) {
    if (singleSectorNum != -1 && abs(singleSectorNum) != (int)iSector && abs(singleSectorNum) != (int)iSector - 12)
      continue;
    hOccME11a->Fill(occME11a[iSector]);
    hOccME11b->Fill(occME11b[iSector]);
    hOccME12->Fill(occME12[iSector]);
    hOccME13->Fill(occME13[iSector]);
    if (iSector < 12) {
      hOccME21->Fill(occME21[iSector]);
      hOccME22->Fill(occME22[iSector]);
      hOccME31->Fill(occME31[iSector]);
      hOccME32->Fill(occME32[iSector]);
      hOccME41->Fill(occME41[iSector]);
      hOccME42->Fill(occME42[iSector]);
    }
  }
  hOccME42SingleSector->Fill(occME42SingleSector);

  for (unsigned iChamb = 0; iChamb < 12; iChamb++) {
    for (unsigned iSector = 0; iSector < 12; iSector++) {
      if (singleSectorNum != -1 && abs(singleSectorNum) != (int)iSector && abs(singleSectorNum) != (int)iSector - 12)
        continue;
      hOccME1ChambsSubSec1[iChamb]->Fill(occME1ChamberSubSec1[iSector][iChamb]);
      hOccME1ChambsSubSec2[iChamb]->Fill(occME1ChamberSubSec2[iSector][iChamb]);
      if (iChamb < 9) {
        hOccME2Chambs[iChamb]->Fill(occME2Chamber[iSector][iChamb]);
        hOccME3Chambs[iChamb]->Fill(occME3Chamber[iSector][iChamb]);
        hOccME4Chambs[iChamb]->Fill(occME4Chamber[iSector][iChamb]);
      }
    }
  }

  //Vertex Counting

  if (haveRECO)
    if (vertexColTag.label() != "null") {
      edm::Handle<reco::VertexCollection> vertexCol;
      iEvent.getByLabel(vertexColTag, vertexCol);
      reco::VertexCollection::const_iterator vertex;
      for (vertex = vertexCol->begin(); vertex != vertexCol->end(); vertex++) {
        if (vertex->isValid() && !vertex->isFake())  //&& vertex->normalizedChi2()<10)
          nVertex++;
      }
      //std::cout << "nVertex: " << nVertex << std::endl;
      //std::cout << "luminosity: " <<insLumi << std::endl;
      hNVertex->Fill(nVertex);
    }

  tree->Fill();
}

// ------------ method called once each job just before starting event loop  ------------
void LCTOccupancies::beginJob() {
  hNVertex = fs->make<TH1F>("NVertex", "N Primary Vertices", 30, 0, 30);
  hNVertex->GetXaxis()->SetTitle("N Primary Vertices");
  hNVertex->GetYaxis()->SetTitle("Counts");
  hInsLumi = fs->make<TH1F>("InsLumi", "Lumi Section Instantanious Luminosity", 10000, 0, 1000000);
  hInsLumi->GetXaxis()->SetTitle("Lumi Section Instantanious Luminosity (Uncorrected HF--10^{30} cm^{2}s^{-1})");
  hInsLumi->GetYaxis()->SetTitle("Counts");

  hMPCLink = fs->make<TH1F>("MPCLink", "Stub MPC Link Number", 5, -1, 4);
  hMPCLink->GetXaxis()->SetTitle("MPC Link");
  hMPCLink->GetYaxis()->SetTitle("Counts");
  hLocalPhi = fs->make<TH1F>("LocalPhi", "Stub Local #phi", 4096, 0, 2.0 * 62.0 * M_PI / 180.0);  //62.0*M_PI/180.0
  hLocalPhi->GetXaxis()->SetTitle("Local #phi");
  hLocalPhi->GetYaxis()->SetTitle("Counts");
  hPhi = fs->make<TH1F>("Phi", "Stub #phi", 100, 0, 3.2);
  hPhi->GetXaxis()->SetTitle("#phi");
  hPhi->GetYaxis()->SetTitle("Counts");
  hPhiPacked = fs->make<TH1F>("PhiPacked", "Stub #phi Packed", 4096, 0, 4096);  //4096
  hPhiPacked->GetXaxis()->SetTitle("#phi Packed");
  hPhiPacked->GetYaxis()->SetTitle("Counts");
  hEta = fs->make<TH1F>("Eta", "Stub |#eta|", 128, 0.9, 2.4);
  hEta->GetXaxis()->SetTitle("|#eta|");
  hEta->GetYaxis()->SetTitle("Counts");
  hEtaPacked = fs->make<TH1F>("EtaPacked", "Stub #eta Packed", 128, 0, 128);  //128
  hEtaPacked->GetXaxis()->SetTitle("#eta Packed");
  hEtaPacked->GetYaxis()->SetTitle("Counts");
  hBx = fs->make<TH1F>("Bx", "Stub Bx Number", 15, 0, 15);
  hBx->GetXaxis()->SetTitle("bx");
  hBx->GetYaxis()->SetTitle("Counts");

  hStation = fs->make<TH1F>("Station", "Stub Station", 9, -4.5, 4.5);
  hStation->GetXaxis()->SetTitle("Station");
  hStation->GetYaxis()->SetTitle("Counts");
  hSector = fs->make<TH1F>("Sector", "Stub Sector", 12, 0.5, 12.5);
  hSector->GetXaxis()->SetTitle("Sector");
  hSector->GetYaxis()->SetTitle("Counts");
  hSubSector = fs->make<TH1F>("SubSector", "Stub SubSector", 3, 0, 3);
  hSubSector->GetXaxis()->SetTitle("SubSector");
  hSubSector->GetYaxis()->SetTitle("Counts");

  hOccStation1SubSec1 = fs->make<TH1F>("OccStation1SubSec1", "Stub Occupancy, Station 1, Subsector 1", 5, -0.5, 4.5);
  hOccStation1SubSec1->GetXaxis()->SetTitle("Stub Occupancy, Station 1, Subsector 1, Summed over Sectors");
  hOccStation1SubSec1->GetYaxis()->SetTitle("Counts");
  hOccStation1SubSec2 = fs->make<TH1F>("OccStation1SubSec2", "Stub Occupancy, Station 1, Subsector 2", 5, -0.5, 4.5);
  hOccStation1SubSec2->GetXaxis()->SetTitle("Stub Occupancy, Station 1, Subsector 1, Summed over Sectors");
  hOccStation1SubSec2->GetYaxis()->SetTitle("Counts");
  hOccStation2 = fs->make<TH1F>("OccStation2", "Stub Occupancy, Station 2", 5, -0.5, 4.5);
  hOccStation2->GetXaxis()->SetTitle("Stub Occupancy, Station 2, Summed over Sectors");
  hOccStation2->GetYaxis()->SetTitle("Counts");
  hOccStation3 = fs->make<TH1F>("OccStation3", "Stub Occupancy, Station 3", 5, -0.5, 4.5);
  hOccStation3->GetXaxis()->SetTitle("Stub Occupancy, Station 3, Summed over Sectors");
  hOccStation3->GetYaxis()->SetTitle("Counts");
  hOccStation4 = fs->make<TH1F>("OccStation4", "Stub Occupancy, Station 4", 5, -0.5, 4.5);
  hOccStation4->GetXaxis()->SetTitle("Stub Occupancy, Station 4, Summed over Sectors");
  hOccStation4->GetYaxis()->SetTitle("Counts");
  hOccMax = fs->make<TH1F>("OccMax", "Maximum Stub Occupancy", 5, -0.5, 4.5);
  hOccMax->GetXaxis()->SetTitle("Maximum Stub Occupancy of Stations, Sectors, and Subsectors");
  hOccMax->GetYaxis()->SetTitle("Counts");
  hOccMaxNo0 = fs->make<TH1F>("OccMaxNo0", "Maximum Stub Occupancy", 4, 0.5, 4.5);
  hOccMaxNo0->GetXaxis()->SetTitle("Maximum Stub Occupancy of Stations, Sectors, and Subsectors");
  hOccMaxNo0->GetYaxis()->SetTitle("Counts");
  hStubsTotal = fs->make<TH1F>("StubsTotal", "N Stubs", 20, 0, 20);
  hStubsTotal->GetXaxis()->SetTitle("N Stubs Unpacked in Event");
  hStubsTotal->GetYaxis()->SetTitle("Counts");

  hOccME11a = fs->make<TH1F>("OccME11a", "Stub Occupancy, ME11a, Summed over Sectors, Subsectors", 5, -0.5, 4.5);
  hOccME11a->GetXaxis()->SetTitle("Stub Occupancy, ME11a, Summed over Sectors, Subsectors");
  hOccME11a->GetYaxis()->SetTitle("Counts");
  hOccME11b = fs->make<TH1F>("OccME11b", "Stub Occupancy, ME11b, Summed over Sectors, Subsectors", 5, -0.5, 4.5);
  hOccME11b->GetXaxis()->SetTitle("Stub Occupancy, ME11b, Summed over Sectors, Subsectors");
  hOccME11b->GetYaxis()->SetTitle("Counts");
  hOccME12 = fs->make<TH1F>("OccME12", "Stub Occupancy, ME12, Summed over Sectors, Subsectors", 5, -0.5, 4.5);
  hOccME12->GetXaxis()->SetTitle("Stub Occupancy, ME12, Summed over Sectors, Subsectors");
  hOccME12->GetYaxis()->SetTitle("Counts");
  hOccME13 = fs->make<TH1F>("OccME13", "Stub Occupancy, ME13, Summed over Sectors, Subsectors", 5, -0.5, 4.5);
  hOccME13->GetXaxis()->SetTitle("Stub Occupancy, ME13, Summed over Sectors, Subsectors");
  hOccME13->GetYaxis()->SetTitle("Counts");
  hOccME21 = fs->make<TH1F>("OccME21", "Stub Occupancy, ME21, Summed over Sectors", 5, -0.5, 4.5);
  hOccME21->GetXaxis()->SetTitle("Stub Occupancy, ME21, Summed over Sectors");
  hOccME21->GetYaxis()->SetTitle("Counts");
  hOccME22 = fs->make<TH1F>("OccME22", "Stub Occupancy, ME22, Summed over Sectors", 5, -0.5, 4.5);
  hOccME22->GetXaxis()->SetTitle("Stub Occupancy, ME22, Summed over Sectors");
  hOccME22->GetYaxis()->SetTitle("Counts");
  hOccME31 = fs->make<TH1F>("OccME31", "Stub Occupancy, ME31, Summed over Sectors", 5, -0.5, 4.5);
  hOccME31->GetXaxis()->SetTitle("Stub Occupancy, ME31, Summed over Sectors");
  hOccME31->GetYaxis()->SetTitle("Counts");
  hOccME32 = fs->make<TH1F>("OccME32", "Stub Occupancy, ME32, Summed over Sectors", 5, -0.5, 4.5);
  hOccME32->GetXaxis()->SetTitle("Stub Occupancy, ME32, Summed over Sectors");
  hOccME32->GetYaxis()->SetTitle("Counts");
  hOccME41 = fs->make<TH1F>("OccME41", "Stub Occupancy, ME41, Summed over Sectors", 5, -0.5, 4.5);
  hOccME41->GetXaxis()->SetTitle("Stub Occupancy, ME41, Summed over Sectors");
  hOccME41->GetYaxis()->SetTitle("Counts");
  hOccME42 = fs->make<TH1F>("OccME42", "Stub Occupancy, ME42, Summed over Sectors", 5, -0.5, 4.5);
  hOccME42->GetXaxis()->SetTitle("Stub Occupancy, ME42, Summed over Sectors");
  hOccME42->GetYaxis()->SetTitle("Counts");
  hOccME42SingleSector =
      fs->make<TH1F>("OccME42SingleSector", "Stub Occupancy, ME42, All Stubs in 1 Sector", 5, -0.5, 4.5);
  hOccME42SingleSector->GetXaxis()->SetTitle("Stub Occupancy, ME42");
  hOccME42SingleSector->GetYaxis()->SetTitle("Counts");

  // Now begin individual chamber modeling
  TFileDirectory chambsME1Dir = fs->mkdir("chambsME1");
  TFileDirectory chambsME2Dir = fs->mkdir("chambsME2");
  TFileDirectory chambsME3Dir = fs->mkdir("chambsME3");
  TFileDirectory chambsME4Dir = fs->mkdir("chambsME4");

  for (int i = 0; i < 12; i++) {
    std::stringstream tmpName;
    std::stringstream tmpTitle;
    tmpName << "OccME1Chamb" << i + 1;
    tmpTitle << "Stub Occupancy ME1, Chamber " << i + 1 << " Summed Over Sectors, SubSectors";
    TH1F* tempHist = chambsME1Dir.make<TH1F>(tmpName.str().c_str(), tmpTitle.str().c_str(), 3, -0.5, 2.5);
    tempHist->GetXaxis()->SetTitle(tmpTitle.str().c_str());
    tempHist->GetYaxis()->SetTitle("Counts");
    hOccME1ChambsSubSec1.push_back(tempHist);
    hOccME1ChambsSubSec2.push_back((TH1F*)tempHist->Clone());
  }

  for (int i = 0; i < 9; i++) {
    for (int j = 2; j < 5; j++) {
      std::stringstream tmpName;
      std::stringstream tmpTitle;
      tmpName << "OccME" << j << "Chamb" << i + 1;
      tmpTitle << "Stub Occupancy ME" << j << ", Chamber " << i + 1 << " Summed Over Sectors";
      TH1F* tempHist;
      if (j == 2) {
        tempHist = chambsME2Dir.make<TH1F>(tmpName.str().c_str(), tmpTitle.str().c_str(), 3, -0.5, 2.5);
        hOccME2Chambs.push_back(tempHist);
      } else if (j == 3) {
        tempHist = chambsME3Dir.make<TH1F>(tmpName.str().c_str(), tmpTitle.str().c_str(), 3, -0.5, 2.5);
        hOccME3Chambs.push_back(tempHist);
      } else if (j == 4) {
        tempHist = chambsME4Dir.make<TH1F>(tmpName.str().c_str(), tmpTitle.str().c_str(), 3, -0.5, 2.5);
        hOccME4Chambs.push_back(tempHist);
      } else
        std::cout << "Warning: problem in initializing Chamber Occ hists!" << std::endl;

      tempHist->GetXaxis()->SetTitle(tmpTitle.str().c_str());
      tempHist->GetYaxis()->SetTitle("Counts");
    }
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void LCTOccupancies::endJob() {
  TH1F* hOccStation1AddSubSec = fs->make<TH1F>("OccStation1AddSubSec", "Stub Occupancy, Station 1", 5, -0.5, 4.5);
  hOccStation1AddSubSec->GetXaxis()->SetTitle("Stub Occupancy, Station 1, Summed over Sectors, Subsectors");
  hOccStation1AddSubSec->GetYaxis()->SetTitle("Counts");
  TH1F* hOccAddStations = fs->make<TH1F>("OccAddStation", "Stub Occupancy", 5, -0.5, 4.5);
  hOccAddStations->GetXaxis()->SetTitle("Stub Occupancy, Summed over Stations, Sectors, Subsectors");
  hOccAddStations->GetYaxis()->SetTitle("Counts");

  hOccStation1AddSubSec->Add(hOccStation1SubSec1);
  hOccStation1AddSubSec->Add(hOccStation1SubSec2);
  hOccAddStations->Add(hOccStation1SubSec1);
  hOccAddStations->Add(hOccStation1SubSec2);
  hOccAddStations->Add(hOccStation2);
  hOccAddStations->Add(hOccStation3);
  hOccAddStations->Add(hOccStation4);
}

void LCTOccupancies::beginLuminosityBlock(edm::LuminosityBlock const& iLumiBlock, edm::EventSetup const& iSetup) {
  if (haveRECO) {
    edm::Handle<LumiSummary> lumiSummary;
    iLumiBlock.getByLabel("lumiProducer", lumiSummary);

    //
    //collect lumi.
    //
    insLumi = lumiSummary->avgInsDelLumi();  //*93.244;
    //std::cout << "luminosity: " <<insLumi << std::endl;
    hInsLumi->Fill(insLumi);
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(LCTOccupancies);
