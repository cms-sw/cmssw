// -*- C++ -*-

// Package:    PixelLumiDQM
// Class:      PixelLumiDQM

/**\class PixelLumiDQM PixelLumiDQM.h PixelLumi/PixelLumiDQM/plugins/PixelLumiDQM.h

 Description: DQM Module producing Pixel Cluster Count Luminosity

 Implementation notes:
     1) Filling scheme is put in by 'hand' for now.
        Can obtain it from the cluster count but need higher rate in trigger; 
	Best would be to have filling scheme in the DB. 
     2) Afterglow correction is put in by hand.
     3) Currently barrel layer 0 is excluded, but a version using all pixel layers and disks is also in place.
     4) A stable beam flag should be obtained from the DB to turn on actual checks. 
        (Pixel Lumi does not make sense outside stable beams.)
     5) Need a top module to correlate the info from the three trigger categories.
        NB: at present the module only uses the ZeroBias trigger providing about 85 Hz.
*/

// Original author:   Amita Raval

#ifndef __PixelLumi_PixelLumiDQM_PixelLumiDQM_h__
#define __PixelLumi_PixelLumiDQM_PixelLumiDQM_h__

#include <vector>
#include <map>
#include <string>
#include <fstream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"

class ConfigurationDescriptions;

class PixelLumiDQM : public DQMEDAnalyzer {
public:
  explicit PixelLumiDQM(const edm::ParameterSet&);
  ~PixelLumiDQM();
  static constexpr double FREQ_ORBIT = 11245.5;
  static constexpr double SECONDS_PER_LS = double(0x40000)/double(FREQ_ORBIT);

  // Using all pixel clusters:
  static constexpr double XSEC_PIXEL_CLUSTER = 10.08e-24; //in cm^2
  static constexpr double XSEC_PIXEL_CLUSTER_UNC = 0.17e-24;

  // Excluding the inner barrel layer.
  static constexpr double rXSEC_PIXEL_CLUSTER = 9.4e-24; //in cm^2
  static constexpr double rXSEC_PIXEL_CLUSTER_UNC = 0.119e-24;
  static constexpr double CM2_TO_NANOBARN = 1.0/1.e-33;
  static const unsigned int lastBunchCrossing = 3564;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const&) override;
  virtual void dqmBeginRun(edm::Run const&, edm::EventSetup const&);
  virtual void endRun(edm::Run const&, edm::EventSetup const&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&,
                                    edm::EventSetup const&);
  virtual void endLuminosityBlock(edm::LuminosityBlock const&,
                                  edm::EventSetup const&);

  // This is a kludge method to infer the filled bunches from the cluster count;  
  // notice that this cannot be used with random triggers.
  // The filling scheme should be acquired from the database once it's there.

  unsigned int calculateBunchMask(MonitorElement *, std::vector<bool> &);
  unsigned int calculateBunchMask(std::vector<float> &, unsigned int, std::vector<bool> &);
  // ---------- Member data ----------

  // Hard-coded numbers of layers and disks...
  static size_t const kNumLayers = 3;
  static size_t const kNumDisks = 2;

  class PixelClusterCount {
    // B for barrel, F for forwared, M for minus, P for plus side, 
    // O for outer and I for inner. Numbers used for layers.
  public:
    PixelClusterCount() :
      numB(kNumLayers, 0),
      numFM(kNumDisks, 0),
      numFP(kNumDisks, 0),
      dnumB(kNumLayers, 0.0),
      dnumFM(kNumDisks, 0.0),
      dnumFP(kNumDisks, 0.0)

    {
    }
    void Reset()
    {
      for(unsigned int i  = 0 ; i< numB.size(); i++){
        numB[i]=0;
        dnumB[i]=0.0;
      }
      for(unsigned int i  = 0 ; i< numFM.size(); i++){
        numFM[i] = 0;
        numFP[i] = 0;
        dnumFM[i] = 0.0;
        dnumFP[i] = 0.0;
      }
    }
    std::vector<UInt_t> numB;
    std::vector<UInt_t> numFM;
    std::vector<UInt_t> numFP;
    std::vector<double> dnumB;
    std::vector<double> dnumFM;
    std::vector<double> dnumFP;
  private:
  };

  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster> > fPixelClusterLabel;

  UInt_t fRunNo;
  UInt_t fEvtNo;
  UInt_t fLSNo;
  UInt_t fBXNo;
  UInt_t fTimestamp;
  UInt_t fNumVtx;
  UInt_t fNumVtxGood;
  UInt_t fNumTrkPerVtx;
  Double_t fVtxX;
  Double_t fVtxY;
  Double_t fVtxZ;
  Double_t fChi2;
  Double_t fNdof;
  std::map<int, PixelClusterCount> fNumPixelClusters;

  bool fIncludeVertexInfo;
  bool fIncludePixelClusterInfo;
  bool fIncludePixelQualCheckHistos;
  int  fResetIntervalInLumiSections;

  std::map<std::string, MonitorElement*> fHistContainerThisRun;

  // This is a list of modules to be ignored, either because they were
  // dead or are dead in part of the data-taking period.
  std::vector<uint32_t> fDeadModules;

  // The minimum number of pixels that should live in a cluster in
  // order for the cluster to be counted as a real cluster 
  // (as opposed to, e.g., a noise pixel).
  int fMinPixelsPerCluster;

  // The minimum pixel cluster charge for a cluster to be counted.
  double fMinClusterCharge;

  // Use the module label to distinguish the three different streams. 

  MonitorElement *fIntActiveCrossingsFromDB;
  MonitorElement *fHistnBClusVsLS[3];
  MonitorElement *fHistnFPClusVsLS[2];
  MonitorElement *fHistnFMClusVsLS[2];
  MonitorElement *fHistBunchCrossings;
  MonitorElement *fHistBunchCrossingsLastLumi;
  MonitorElement *fHistClusterCountByBxLastLumi;
  MonitorElement *fHistClusterCountByBxCumulative;
  MonitorElement *fHistClusByLS;
  MonitorElement *fHistTotalRecordedLumiByLS;
  MonitorElement *fHistRecordedByBxLastLumi;
  MonitorElement *fHistRecordedByBxCumulative;

  std::vector<bool> bunchTriggerMask;
  unsigned int filledAndUnmaskedBunches;
  bool useInnerBarrelLayer;
  unsigned int fFillNumber;
  std::string  fLogFileName_;

  std::ofstream logFile_;
};

#endif
