#ifndef SiPixelMonitorDigi_SiPixelDigiSource_h
#define SiPixelMonitorDigi_SiPixelDigiSource_h
// -*- C++ -*-
//
// Package:     SiPixelMonitorDigi
// Class  :     SiPixelDigiSource
//
/**

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Vincenzo Chiochia
//         Created:
//

#include <memory>

// user include files
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "DQM/SiPixelMonitorDigi/interface/SiPixelDigiModule.h"
#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include <cstdint>

struct SiPixelDigiCounter {
  int nBPIXDigis = 0;
  int nFPIXDigis = 0;
  int nDigisPerFed[40];
  int NzeroROCs[2];
  int NloEffROCs[2];

  bool ROCMapToReset = false;

  bool DoZeroRocsBMO1 = false;
  bool DoZeroRocsBMO2 = false;
  bool DoZeroRocsBMO3 = false;

  bool DoZeroRocsBMI1 = false;
  bool DoZeroRocsBMI2 = false;
  bool DoZeroRocsBMI3 = false;

  bool DoZeroRocsBPO1 = false;
  bool DoZeroRocsBPO2 = false;
  bool DoZeroRocsBPO3 = false;

  bool DoZeroRocsBPI1 = false;
  bool DoZeroRocsBPI2 = false;
  bool DoZeroRocsBPI3 = false;

  bool DoZeroRocsFPO1 = false;
  bool DoZeroRocsFPO2 = false;

  bool DoZeroRocsFMO1 = false;
  bool DoZeroRocsFMO2 = false;

  bool DoZeroRocsFPI1 = false;
  bool DoZeroRocsFPI2 = false;

  bool DoZeroRocsFMI1 = false;
  bool DoZeroRocsFMI2 = false;
  SiPixelDigiCounter() {
    //all initialization which was done in BeginRun
    nBPIXDigis = 0;
    nFPIXDigis = 0;
    for (int i = 0; i != 40; i++)
      nDigisPerFed[i] = 0;
    for (int i = 0; i < 2; i++)
      NzeroROCs[i] = 0;
    for (int i = 0; i < 2; i++)
      NloEffROCs[i] = 0;
  };
};

class SiPixelDigiSource : public DQMOneEDAnalyzer<edm::LuminosityBlockCache<SiPixelDigiCounter>> {
public:
  explicit SiPixelDigiSource(const edm::ParameterSet& conf);
  ~SiPixelDigiSource() override;

  typedef edm::DetSet<PixelDigi>::const_iterator DigiIterator;

  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void dqmBeginRun(const edm::Run&, edm::EventSetup const&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  std::shared_ptr<SiPixelDigiCounter> globalBeginLuminosityBlock(const edm::LuminosityBlock& lumi,
                                                                 const edm::EventSetup& iSetup) const override;
  void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  virtual void buildStructure(edm::EventSetup const&);
  virtual void bookMEs(DQMStore::IBooker&, const edm::EventSetup& iSetup);

  virtual void CountZeroROCsInSubstructure(bool, bool&, SiPixelDigiModule*, SiPixelDigiCounter*);

  std::string topFolderName_;

private:
  edm::ParameterSet conf_;
  edm::InputTag src_;
  bool saveFile;
  bool isPIB;
  bool slowDown;
  bool modOn;
  bool twoDimOn;
  bool twoDimModOn;
  bool twoDimOnlyLayDisk;
  bool hiRes;
  bool reducedSet;
  //barrel:
  bool ladOn, layOn, phiOn;
  //forward:
  bool ringOn, bladeOn, diskOn;
  std::map<uint32_t, SiPixelDigiModule*> thePixelStructure;

  int nDP1P1M1;
  int nDP1P1M2;
  int nDP1P1M3;
  int nDP1P1M4;
  int nDP1P2M1;
  int nDP1P2M2;
  int nDP1P2M3;
  int nDP2P1M1;
  int nDP2P1M2;
  int nDP2P1M3;
  int nDP2P1M4;
  int nDP2P2M1;
  int nDP2P2M2;
  int nDP2P2M3;
  int nDP3P1M1;
  int nDP3P2M1;
  int nDM1P1M1;
  int nDM1P1M2;
  int nDM1P1M3;
  int nDM1P1M4;
  int nDM1P2M1;
  int nDM1P2M2;
  int nDM1P2M3;
  int nDM2P1M1;
  int nDM2P1M2;
  int nDM2P1M3;
  int nDM2P1M4;
  int nDM2P2M1;
  int nDM2P2M2;
  int nDM2P2M3;
  int nDM3P1M1;
  int nDM3P2M1;
  int nL1M1;
  int nL1M2;
  int nL1M3;
  int nL1M4;
  int nL2M1;
  int nL2M2;
  int nL2M3;
  int nL2M4;
  int nL3M1;
  int nL3M2;
  int nL3M3;
  int nL3M4;
  int nL4M1;
  int nL4M2;
  int nL4M3;
  int nL4M4;
  int nBigEvents;

  MonitorElement* bigEventRate;
  MonitorElement* pixEvtsPerBX;
  MonitorElement* pixEventRate;
  MonitorElement* noOccROCsBarrel;
  MonitorElement* loOccROCsBarrel;
  MonitorElement* noOccROCsEndcap;
  MonitorElement* loOccROCsEndcap;
  MonitorElement* averageDigiOccupancy;
  MonitorElement* avgBarrelFedOccvsLumi;
  MonitorElement* avgEndcapFedOccvsLumi;
  MonitorElement* avgfedDigiOccvsLumi;
  MonitorElement* meNDigisCOMBBarrel_;
  MonitorElement* meNDigisCOMBEndcap_;
  MonitorElement* meNDigisCHANBarrel_;
  std::vector<MonitorElement*> meNDigisCHANBarrelLs_;
  MonitorElement* meNDigisCHANBarrelCh1_;
  MonitorElement* meNDigisCHANBarrelCh2_;
  MonitorElement* meNDigisCHANBarrelCh3_;
  MonitorElement* meNDigisCHANBarrelCh4_;
  MonitorElement* meNDigisCHANBarrelCh5_;
  MonitorElement* meNDigisCHANBarrelCh6_;
  MonitorElement* meNDigisCHANBarrelCh7_;
  MonitorElement* meNDigisCHANBarrelCh8_;
  MonitorElement* meNDigisCHANBarrelCh9_;
  MonitorElement* meNDigisCHANBarrelCh10_;
  MonitorElement* meNDigisCHANBarrelCh11_;
  MonitorElement* meNDigisCHANBarrelCh12_;
  MonitorElement* meNDigisCHANBarrelCh13_;
  MonitorElement* meNDigisCHANBarrelCh14_;
  MonitorElement* meNDigisCHANBarrelCh15_;
  MonitorElement* meNDigisCHANBarrelCh16_;
  MonitorElement* meNDigisCHANBarrelCh17_;
  MonitorElement* meNDigisCHANBarrelCh18_;
  MonitorElement* meNDigisCHANBarrelCh19_;
  MonitorElement* meNDigisCHANBarrelCh20_;
  MonitorElement* meNDigisCHANBarrelCh21_;
  MonitorElement* meNDigisCHANBarrelCh22_;
  MonitorElement* meNDigisCHANBarrelCh23_;
  MonitorElement* meNDigisCHANBarrelCh24_;
  MonitorElement* meNDigisCHANBarrelCh25_;
  MonitorElement* meNDigisCHANBarrelCh26_;
  MonitorElement* meNDigisCHANBarrelCh27_;
  MonitorElement* meNDigisCHANBarrelCh28_;
  MonitorElement* meNDigisCHANBarrelCh29_;
  MonitorElement* meNDigisCHANBarrelCh30_;
  MonitorElement* meNDigisCHANBarrelCh31_;
  MonitorElement* meNDigisCHANBarrelCh32_;
  MonitorElement* meNDigisCHANBarrelCh33_;
  MonitorElement* meNDigisCHANBarrelCh34_;
  MonitorElement* meNDigisCHANBarrelCh35_;
  MonitorElement* meNDigisCHANBarrelCh36_;
  MonitorElement* meNDigisCHANEndcap_;
  std::vector<MonitorElement*> meNDigisCHANEndcapDps_;
  std::vector<MonitorElement*> meNDigisCHANEndcapDms_;

  //the following long list of bools is to patch the ZeroOccupancy ROC filling in a way that a substructure (like BPix/BmO/Layer1) is counted only once as it should be (in the past for each module in the substructure the same number of ZeroOccupancy rocs was added)

  int bigEventSize;
  bool isUpgrade;
  bool firstRun;

  std::string I_name[1856];
  unsigned int I_detId[1856];
  int I_fedId[1856];
  int I_linkId1[1856];
  int I_linkId2[1856];

  int nDigisPerChan[1152];
  int nDigisPerDisk[6];
  int numberOfDigis[336];
  int nDigisA;
  int nDigisB;
  //define Token(-s)
  edm::EDGetTokenT<edm::DetSetVector<PixelDigi>> srcToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopoToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> trackerTopoTokenBeginRun_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> trackerGeomTokenBeginRun_;
  int noOfLayers;
  int noOfDisks;
};

#endif
