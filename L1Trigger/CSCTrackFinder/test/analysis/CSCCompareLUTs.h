#ifndef CSCTrackFinder_CompareLUTs_h
#define CSCTrackFinder_CompareLUTs_h

/**
 * \author B. Jackson 2/24/08
 *
 */

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <iostream>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCTFPtLUT.h"

#include <TFile.h>
#include <TH1I.h>
#include <TH2I.h>

class CSCSectorReceiverLUT;

class CSCCompareLUTs : public edm::EDAnalyzer {
 public:
  explicit CSCCompareLUTs(edm::ParameterSet const& conf);
  virtual ~CSCCompareLUTs();
  virtual void endJob();
  virtual void analyze(edm::Event const& e, edm::EventSetup const& iSetup);

  /// Helpers
  std::string encodeFileIndex(int _endcap, int _sector, int _station, int _subsector) const; // ME 1 with subsectors
  std::string encodeFileIndex(int _endcap, int _sector, int _station) const; // ME 234
  
 private:
  // method to actually compare LUTs  
  void compareLUTs(CSCSectorReceiverLUT *SRLUT_1, CSCSectorReceiverLUT *SRLUT_2, CSCTFPtLUT *PtLUT_1, CSCTFPtLUT *PtLUT_2,
		   bool doLocalPhi, bool doGlobalEta, bool doGlobalPhiME, bool doGlobalPhiMB, bool DoPt,
		   const int _endcap, const int _sector, const int _subsector, const int _station);

  // returns the value for global eta
  double getGlobalEtaValue(const int _endcap, const int _sector, const int _subsector, const int _station,
			   const unsigned& cscid, const unsigned& wire_group, const unsigned& phi_local) const;
  
  // variables persistent across events should be declared here.
  //
  CSCSectorReceiverLUT *SRLUT_Base_1, *SRLUT_Base_2; // [Endcap][Sector][Subsector][Station]
  CSCTFPtLUT *PtLUT_Base_1, *PtLUT_Base_2;
  int endcap, sector, station, subsector, isTMB07;
  edm::ParameterSet lutParam1, lutParam2;
  
  TFile* fCompare;

  std::string outFileName, lut1name, lut2name;

  // 2d plots that compare the output words for all addresses
  TH2I* compareLocalPhi;
  TH2I* compareLocalPhi_phiLocal;
  TH2I* compareLocalPhi_phiBend;
  
  TH2I* compareGlobalEta;
  TH2I* compareGlobalEta_etaGlobal;
  TH2I* compareGlobalEta_phiBend;
  
  TH2I* compareGlobalPhiME;
  
  TH2I* compareGlobalPhiMB;
  
  TH2I* comparePt;
  TH2I* comparePt_front;
  TH2I* comparePt_rear;
  
  TH2I* compareLocalPhiOffDiagonal;
  TH2I* compareLocalPhiOffDiagonal_phiLocal;
  TH2I* compareLocalPhiOffDiagonal_phiBend;
  
  TH2I* compareGlobalEtaOffDiagonal;
  TH2I* compareGlobalEtaOffDiagonal_etaGlobal;
  TH2I* compareGlobalEtaOffDiagonal_phiBend;
  
  TH2I* compareGlobalPhiMEOffDiagonal;
  
  TH2I* compareGlobalPhiMBOffDiagonal;
  
  TH2I* comparePtOffDiagonal;
  TH2I* comparePtOffDiagonal_front;
  TH2I* comparePtOffDiagonal_rear;
  
  TH1I* differenceLocalPhi;
  TH1I* differenceLocalPhi_phiLocal;
  TH1I* differenceLocalPhi_phiBend;
  
  TH1I* differenceGlobalEta;
  TH1I* differenceGlobalEta_etaGlobal;
  TH1I* differenceGlobalEta_phiBend;
  
  TH1I* differenceGlobalPhiME;
  
  TH1I* differenceGlobalPhiMB;
  
  TH1I* differencePt;
  TH1I* differencePt_front;
  TH1I* differencePt_rear;
  
  // When the output word does not match, adds the address to these histograms
  TH1I* mismatchLocalPhiAddress;
  TH1I* mismatchLocalPhiAddress_patternId;
  TH1I* mismatchLocalPhiAddress_patternNumber;
  TH1I* mismatchLocalPhiAddress_quality;
  TH1I* mismatchLocalPhiAddress_leftRight;
  TH1I* mismatchLocalPhiAddress_spare;
  
  TH1I* mismatchGlobalEtaAddress;
  TH1I* mismatchGlobalEtaAddress_phiBendLocal;
  TH1I* mismatchGlobalEtaAddress_phiLocal;
  TH1I* mismatchGlobalEtaAddress_wireGroup;
  TH1I* mismatchGlobalEtaAddress_cscId;

  TH1I* mismatchGlobalPhiMEAddress;
  TH1I* mismatchGlobalPhiMEAddress_phiLocal;
  TH1I* mismatchGlobalPhiMEAddress_wireGroup;
  TH1I* mismatchGlobalPhiMEAddress_cscId;

  TH1I* mismatchGlobalPhiMBAddress;
  TH1I* mismatchGlobalPhiMBAddress_phiLocal;
  TH1I* mismatchGlobalPhiMBAddress_wireGroup;
  TH1I* mismatchGlobalPhiMBAddress_cscId;

  TH1I* mismatchPtAddress;
  TH1I* mismatchPtAddress_delta12phi;
  TH1I* mismatchPtAddress_delta23phi;
  TH1I* mismatchPtAddress_deltaPhi;
  TH1I* mismatchPtAddress_eta;
  TH1I* mismatchPtAddress_mode;
  TH1I* mismatchPtAddress_sign;


  // address versus output word for LUT 1
  TH2I* InputVsOutputLocalPhi_1;
  TH2I* InputVsOutputGlobalEta_1;
  TH2I* InputVsOutputGlobalPhiME_1;
  TH2I* InputVsOutputGlobalPhiMB_1;
  TH2I* InputVsOutputPt_1;

  // address versus output word for LUT 2
  TH2I* InputVsOutputLocalPhi_2;
  TH2I* InputVsOutputGlobalEta_2;
  TH2I* InputVsOutputGlobalPhiME_2;
  TH2I* InputVsOutputGlobalPhiMB_2;
  TH2I* InputVsOutputPt_2;

  
};

DEFINE_FWK_MODULE(CSCCompareLUTs);

#endif
