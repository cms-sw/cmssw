/**
 * Analyzer to compare one LUT to another and record the differences.
 */

#include <fstream>
#include <iostream>
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include <Geometry/Records/interface/MuonGeometryRecord.h>

#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"

#include "DataFormats/L1CSCTrackFinder/interface/CSCBitWidths.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCSectorReceiverLUT.h"
#include "L1Trigger/CSCTrackFinder/interface/CSCTFPtLUT.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h"
#include "DataFormats/L1CSCTrackFinder/interface/CSCTFConstants.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"

#include <FWCore/MessageLogger/interface/MessageLogger.h>

#include "L1Trigger/CSCTrackFinder/test/analysis/CSCCompareLUTs.h"

#include "CondFormats/L1TObjects/interface/L1MuTriggerScales.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerScalesRcd.h"
#include "CondFormats/L1TObjects/interface/L1MuTriggerPtScale.h"
#include "CondFormats/DataRecord/interface/L1MuTriggerPtScaleRcd.h"

CSCCompareLUTs::CSCCompareLUTs(edm::ParameterSet const& conf)
{
  station = conf.getUntrackedParameter<int>("Station",-1);
  sector = conf.getUntrackedParameter<int>("Sector",-1);
  subsector = conf.getUntrackedParameter<int>("SubSector",-1);
  endcap = conf.getUntrackedParameter<int>("Endcap",-1);
  lutParam1 = conf.getParameter<edm::ParameterSet>("lutParam1");
  lutParam2 = conf.getParameter<edm::ParameterSet>("lutParam2");
  outFileName = conf.getUntrackedParameter<std::string>("OutFileName", "LUT_compare.root");
  
  lut1name = lutParam1.getUntrackedParameter<std::string>("lutName", "LUT 1");
  lut2name = lutParam2.getUntrackedParameter<std::string>("lutName", "LUT 2");
  
  fCompare = new TFile(TString(outFileName), "RECREATE");
  
  compareLocalPhi = new TH2I("compareLocalPhi", "Local Phi Data Field Comparison",
                             1<<int((CSCBitWidths::kLocalPhiDataBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth)/2),
                             0, 1<<(CSCBitWidths::kLocalPhiDataBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth),
                             1<<int((CSCBitWidths::kLocalPhiDataBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth)/2),
                             0, 1<<(CSCBitWidths::kLocalPhiDataBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth) );
  compareLocalPhi->GetXaxis()->SetTitle(TString(lut1name));
  compareLocalPhi->GetYaxis()->SetTitle(TString(lut2name));
  
  compareLocalPhi_phiLocal = new TH2I("compareLocalPhi_phiLocal", "Local Phi Data Field Comparison (Phi Local Word)",
                                      1<<CSCBitWidths::kLocalPhiDataBitWidth,
                                      0, 1<<CSCBitWidths::kLocalPhiDataBitWidth,
                                      1<<CSCBitWidths::kLocalPhiDataBitWidth,
                                      0, 1<<CSCBitWidths::kLocalPhiDataBitWidth );
  compareLocalPhi_phiLocal->GetXaxis()->SetTitle(TString(lut1name));
  compareLocalPhi_phiLocal->GetYaxis()->SetTitle(TString(lut2name));
  
  compareLocalPhi_phiBend = new TH2I("compareLocalPhi_phiBend", "Local Phi Data Field Comparison (Phi Bend Word)",
                                     1<<CSCBitWidths::kLocalPhiBendDataBitWidth,
                                     0, 1<<CSCBitWidths::kLocalPhiBendDataBitWidth,
                                     1<<CSCBitWidths::kLocalPhiBendDataBitWidth,
                                     0, 1<<CSCBitWidths::kLocalPhiBendDataBitWidth );
  compareLocalPhi_phiBend->GetXaxis()->SetTitle(TString(lut1name));
  compareLocalPhi_phiBend->GetYaxis()->SetTitle(TString(lut2name));
  
  compareGlobalEta = new TH2I("compareGlobalEta", "Global Eta Data Field Comparison",
                              1<<(CSCBitWidths::kGlobalEtaBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth - 1),
                              0, 1<<(CSCBitWidths::kGlobalEtaBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth - 1),
                              1<<(CSCBitWidths::kGlobalEtaBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth - 1),
                              0, 1<<(CSCBitWidths::kGlobalEtaBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth - 1) ); 
  compareGlobalEta->GetXaxis()->SetTitle(TString(lut1name));
  compareGlobalEta->GetYaxis()->SetTitle(TString(lut2name));
  
  compareGlobalEta_etaGlobal = new TH2I("compareGlobalEta_etaGlobal", "Global Eta Data Field Comparison (Eta Global Word)",
                                        1<<CSCBitWidths::kGlobalEtaBitWidth,
                                        0, 1<<CSCBitWidths::kGlobalEtaBitWidth,
                                        1<<CSCBitWidths::kGlobalEtaBitWidth,
                                        0, 1<<CSCBitWidths::kGlobalEtaBitWidth ); 
  compareGlobalEta_etaGlobal->GetXaxis()->SetTitle(TString(lut1name));
  compareGlobalEta_etaGlobal->GetYaxis()->SetTitle(TString(lut2name));
  
  compareGlobalEta_phiBend = new TH2I("compareGlobalEta_phiBend", "Global Eta Data Field Comparison (Phi Bend Word)",
                                      1<<(CSCBitWidths::kLocalPhiBendDataBitWidth - 1),
                                      0, 1<<(CSCBitWidths::kLocalPhiBendDataBitWidth - 1),
                                      1<<(CSCBitWidths::kLocalPhiBendDataBitWidth - 1),
                                      0, 1<<(CSCBitWidths::kLocalPhiBendDataBitWidth - 1) ); 
  compareGlobalEta_phiBend->GetXaxis()->SetTitle(TString(lut1name));
  compareGlobalEta_phiBend->GetYaxis()->SetTitle(TString(lut2name));
  
  compareGlobalPhiME = new TH2I("compareGlobalPhiME", "Global Phi ME Data Field Comparison",
                                1<<CSCBitWidths::kGlobalPhiDataBitWidth,
                                0, 1<<CSCBitWidths::kGlobalPhiDataBitWidth,
                                1<<CSCBitWidths::kGlobalPhiDataBitWidth,
                                0, 1<<CSCBitWidths::kGlobalPhiDataBitWidth );
  compareGlobalPhiME->GetXaxis()->SetTitle(TString(lut1name));
  compareGlobalPhiME->GetYaxis()->SetTitle(TString(lut2name));
  
  compareGlobalPhiMB = new TH2I("compareGlobalPhiMB", "Global Phi MB Data Field Comparison",
                                1<<CSCBitWidths::kGlobalPhiDataBitWidth,
                                0, 1<<CSCBitWidths::kGlobalPhiDataBitWidth,
                                1<<CSCBitWidths::kGlobalPhiDataBitWidth,
                                0, 1<<CSCBitWidths::kGlobalPhiDataBitWidth );
  compareGlobalPhiMB->GetXaxis()->SetTitle(TString(lut1name));
  compareGlobalPhiMB->GetYaxis()->SetTitle(TString(lut2name));
  
  comparePt = new TH2I("comparePt", "Pt Data Field Comparison",
                       1<<int(16/2),
                       0, 1<<16,
                       1<<int(16/2),
                       0, 1<<16 );
  comparePt->GetXaxis()->SetTitle(TString(lut1name));
  comparePt->GetYaxis()->SetTitle(TString(lut2name));
  
  comparePt_front = new TH2I("comparePt_front", "Pt Data Field Comparison (Front)",
                             1<<8,
                             0, 1<<8,
                             1<<8,
                             0, 1<<8 );
  comparePt_front->GetXaxis()->SetTitle(TString(lut1name));
  comparePt_front->GetYaxis()->SetTitle(TString(lut2name));
  
  comparePt_rear = new TH2I("comparePt_rear", "Pt Data Field Comparison (Rear)",
                            1<<8,
                            0, 1<<8,
                            1<<8,
                            0, 1<<8 );
  comparePt_rear->GetXaxis()->SetTitle(TString(lut1name));
  comparePt_rear->GetYaxis()->SetTitle(TString(lut2name));
  
  
  compareLocalPhiOffDiagonal = new TH2I("compareLocalPhiOffDiagonal", "Local Phi Data Field Comparison: Off Diagonal",
                                        1<<int((CSCBitWidths::kLocalPhiDataBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth)/2),
                                        0, 1<<(CSCBitWidths::kLocalPhiDataBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth),
                                        1<<int((CSCBitWidths::kLocalPhiDataBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth)/2),
                                        0, 1<<(CSCBitWidths::kLocalPhiDataBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth) );
  compareLocalPhiOffDiagonal->GetXaxis()->SetTitle(TString(lut1name));
  compareLocalPhiOffDiagonal->GetYaxis()->SetTitle(TString(lut2name));
  
  compareLocalPhiOffDiagonal_phiLocal = new TH2I("compareLocalPhiOffDiagonal_phiLocal", "Local Phi Data Field Comparison: Off Diagonal (Local Phi Word)",
                                                 1<<CSCBitWidths::kLocalPhiDataBitWidth,
                                                 0, 1<<CSCBitWidths::kLocalPhiDataBitWidth,
                                                 1<<CSCBitWidths::kLocalPhiDataBitWidth,
                                                 0, 1<<CSCBitWidths::kLocalPhiDataBitWidth );
  compareLocalPhiOffDiagonal_phiLocal->GetXaxis()->SetTitle(TString(lut1name));
  compareLocalPhiOffDiagonal_phiLocal->GetYaxis()->SetTitle(TString(lut2name));
  
  compareLocalPhiOffDiagonal_phiBend = new TH2I("compareLocalPhiOffDiagonal_phiBend", "Local Phi Data Field Comparison: Off Diagonal (Phi Bend Word)",
                                                1<<CSCBitWidths::kLocalPhiBendDataBitWidth,
                                                0, 1<<CSCBitWidths::kLocalPhiBendDataBitWidth,
                                                1<<CSCBitWidths::kLocalPhiBendDataBitWidth,
                                                0, 1<<CSCBitWidths::kLocalPhiBendDataBitWidth );
  compareLocalPhiOffDiagonal_phiBend->GetXaxis()->SetTitle(TString(lut1name));
  compareLocalPhiOffDiagonal_phiBend->GetYaxis()->SetTitle(TString(lut2name));
  
  compareGlobalEtaOffDiagonal = new TH2I("compareGlobalEtaOffDiagonal", "Global Eta Data Field Comparison: Off Diagonal",
                                         1<<(CSCBitWidths::kGlobalEtaBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth - 1),
                                         0, 1<<(CSCBitWidths::kGlobalEtaBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth - 1),
                                         1<<(CSCBitWidths::kGlobalEtaBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth - 1),
                                         0, 1<<(CSCBitWidths::kGlobalEtaBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth - 1) ); 
  compareGlobalEtaOffDiagonal->GetXaxis()->SetTitle(TString(lut1name));
  compareGlobalEtaOffDiagonal->GetYaxis()->SetTitle(TString(lut2name));
  
  compareGlobalEtaOffDiagonal_etaGlobal = new TH2I("compareGlobalEtaOffDiagonal_etaGlobal", "Global Eta Data Field Comparison: Off Diagonal (Eta Global Word)",
                                                   1<<CSCBitWidths::kGlobalEtaBitWidth,
                                                   0, 1<<CSCBitWidths::kGlobalEtaBitWidth,
                                                   1<<CSCBitWidths::kGlobalEtaBitWidth,
                                                   0, 1<<CSCBitWidths::kGlobalEtaBitWidth ); 
  compareGlobalEtaOffDiagonal_etaGlobal->GetXaxis()->SetTitle(TString(lut1name));
  compareGlobalEtaOffDiagonal_etaGlobal->GetYaxis()->SetTitle(TString(lut2name));
  
  compareGlobalEtaOffDiagonal_phiBend = new TH2I("compareGlobalEtaOffDiagonal_phiBend", "Global Eta Data Field Comparison: Off Diagonal (Phi Bend Word)",
                                                 1<<(CSCBitWidths::kLocalPhiBendDataBitWidth - 1),
                                                 0, 1<<(CSCBitWidths::kLocalPhiBendDataBitWidth - 1),
                                                 1<<(CSCBitWidths::kLocalPhiBendDataBitWidth - 1),
                                                 0, 1<<(CSCBitWidths::kLocalPhiBendDataBitWidth - 1) ); 
  compareGlobalEtaOffDiagonal_phiBend->GetXaxis()->SetTitle(TString(lut1name));
  compareGlobalEtaOffDiagonal_phiBend->GetYaxis()->SetTitle(TString(lut2name));
  
  compareGlobalPhiMEOffDiagonal = new TH2I("compareGlobalPhiMEOffDiagonal", "Global Phi ME Data Field Comparison: Off Diagonal",
                                           1<<int((CSCBitWidths::kGlobalPhiDataBitWidth)),
                                           0, 1<<(CSCBitWidths::kGlobalPhiDataBitWidth),
                                           1<<int((CSCBitWidths::kGlobalPhiDataBitWidth)),
                                           0, 1<<(CSCBitWidths::kGlobalPhiDataBitWidth) );
  compareGlobalPhiMEOffDiagonal->GetXaxis()->SetTitle(TString(lut1name));
  compareGlobalPhiMEOffDiagonal->GetYaxis()->SetTitle(TString(lut2name));
  
  compareGlobalPhiMBOffDiagonal = new TH2I("compareGlobalPhiMBOffDiagonal", "Global Phi MB Data Field Comparison: Off Diagonal",
                                           1<<int((CSCBitWidths::kGlobalPhiDataBitWidth)),
                                           0, 1<<(CSCBitWidths::kGlobalPhiDataBitWidth),
                                           1<<int((CSCBitWidths::kGlobalPhiDataBitWidth)),
                                           0, 1<<(CSCBitWidths::kGlobalPhiDataBitWidth) );
  compareGlobalPhiMBOffDiagonal->GetXaxis()->SetTitle(TString(lut1name));
  compareGlobalPhiMBOffDiagonal->GetYaxis()->SetTitle(TString(lut2name));
  
  comparePtOffDiagonal = new TH2I("comparePtOffDiagonal", "Pt Data Field Comparison: Off Diagonal",
                                  1<<int(16/2),
                                  0, 1<<16,
                                  1<<int(16/2),
                                  0, 1<<16 );
  comparePtOffDiagonal->GetXaxis()->SetTitle(TString(lut1name));
  comparePtOffDiagonal->GetYaxis()->SetTitle(TString(lut2name));
  
  comparePtOffDiagonal_front = new TH2I("comparePtOffDiagonal_front", "Pt Data Field Comparison: Off Diagonal (Front)",
                                        1<<8,
                                        0, 1<<8,
                                        1<<8,
                                        0, 1<<8 );
  comparePtOffDiagonal_front->GetXaxis()->SetTitle(TString(lut1name));
  comparePtOffDiagonal_front->GetYaxis()->SetTitle(TString(lut2name));
  
  comparePtOffDiagonal_rear = new TH2I("comparePtOffDiagonal_rear", "Pt Data Field Comparison: Off Diagonal (Rear)",
                                       1<<8,
                                       0, 1<<8,
                                       1<<8,
                                       0, 1<<8 );
  comparePtOffDiagonal_rear->GetXaxis()->SetTitle(TString(lut1name));
  comparePtOffDiagonal_rear->GetYaxis()->SetTitle(TString(lut2name));
  
  
  differenceLocalPhi = new TH1I("differenceLocalPhi", "Local Phi: Difference in Data Field",
                                2<<(CSCBitWidths::kLocalPhiDataBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth),
                                -(1<<(CSCBitWidths::kLocalPhiDataBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth)), 
                                1<<(CSCBitWidths::kLocalPhiDataBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth));
  differenceLocalPhi_phiLocal = new TH1I("differenceLocalPhi_phiLocal", "Local Phi: Difference in Data Field (Phi Local Word)",
                                         1<<CSCBitWidths::kLocalPhiDataBitWidth, 
                                         -(1<<CSCBitWidths::kLocalPhiDataBitWidth), 
                                         1<<CSCBitWidths::kLocalPhiDataBitWidth);
  differenceLocalPhi_phiBend = new TH1I("differenceLocalPhi_phiBend", "Local Phi: Difference in Data Field (Phi Bend Word)",
                                        2<<CSCBitWidths::kLocalPhiBendDataBitWidth,
                                        -(1<<CSCBitWidths::kLocalPhiBendDataBitWidth), 
                                        1<<CSCBitWidths::kLocalPhiBendDataBitWidth);
  differenceGlobalEta = new TH1I("differenceGlobalEta", "Global Eta: Difference in Data Field",
                                 2<<(CSCBitWidths::kGlobalEtaBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth),
                                 -(1<<(CSCBitWidths::kGlobalEtaBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth)), 
                                 1<<(CSCBitWidths::kGlobalEtaBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth));
  differenceGlobalEta_etaGlobal = new TH1I("differenceGlobalEta_etaGlobal", "Global Eta: Difference in Data Field (Eta Global Word)",
                                           2<<CSCBitWidths::kGlobalEtaBitWidth,
                                           -(1<<CSCBitWidths::kGlobalEtaBitWidth), 
                                           1<<CSCBitWidths::kGlobalEtaBitWidth);
  differenceGlobalEta_phiBend = new TH1I("differenceGlobalEta_phiBend", "Global Eta: Difference in Data Field (Phi Bend Word)",
                                         2<<(CSCBitWidths::kLocalPhiBendDataBitWidth),
                                         -(1<<(CSCBitWidths::kLocalPhiBendDataBitWidth)), 
                                         1<<(CSCBitWidths::kLocalPhiBendDataBitWidth));
  differenceGlobalPhiME = new TH1I("differenceGlobalPhiME", "Global Phi ME: Difference in Data Field",
                                   2<<CSCBitWidths::kGlobalPhiDataBitWidth,
                                   -(1<<CSCBitWidths::kGlobalPhiDataBitWidth), 
                                   1<<CSCBitWidths::kGlobalPhiDataBitWidth);
  differenceGlobalPhiMB = new TH1I("differenceGlobalPhiMB", "Global Phi MB: Difference in Data Field",
                                   2<<CSCBitWidths::kGlobalPhiDataBitWidth,
                                   -(1<<CSCBitWidths::kGlobalPhiDataBitWidth), 
                                   1<<CSCBitWidths::kGlobalPhiDataBitWidth);
  differencePt = new TH1I("differencePt", "Pt: Difference in Data Field",
                          1<<16,
                          0, 1<<16);
  differencePt_front = new TH1I("differencePt_front", "Pt: Difference in Data Field (Front)",
                                1<<8,
                                0, 1<<8);
  differencePt_rear = new TH1I("differencePt_rear", "Pt: Difference in Data Field (Rear)",
                               1<<8,
                               0, 1<<8);
  
  mismatchLocalPhiAddress = new TH1I("mismatchLocalPhiAddress", "Local Phi Address (data fields do not match)",
                                     1<<int(CSCBitWidths::kLocalPhiAddressWidth/2),
                                     0, 1<<CSCBitWidths::kLocalPhiAddressWidth);
  mismatchLocalPhiAddress_patternId = new TH1I("mismatchLocalPhiAddress_patternId", "Local Phi Address - Pattern ID Word (data fields do not match)", 1<<8, 0, 1<<8);
  mismatchLocalPhiAddress_patternNumber = new TH1I("mismatchLocalPhiAddress_patternNumber", "Local Phi Address (Legal Addresses Only) - Pattern Number Word (data fields do not match)", 1<<4, 0, 1<<4);
  mismatchLocalPhiAddress_quality = new TH1I("mismatchLocalPhiAddress_quality", "Local Phi Address (Legal Addresses Only) - Quality Word (data fields do not match)", 1<<4, 0, 1<<4);
  mismatchLocalPhiAddress_leftRight = new TH1I("mismatchLocalPhiAddress_leftRight", "Local Phi Address (Legal Addresses Only) - Left/Right Word (data fields do not match)", 1<<1, 0, 1<<1);
  mismatchLocalPhiAddress_spare = new TH1I("mismatchLocalPhiAddress_spare", "Local Phi Address (Legal Addresses Only) - Spare Word (data fields do not match)", 1<<2, 0, 1<<2);
  mismatchGlobalEtaAddress = new TH1I("mismatchGlobalEtaAddress", "Global Eta Address (data fields do not match)",
                                      1<<int(CSCBitWidths::kGlobalEtaAddressWidth/2),
                                      0,1<<CSCBitWidths::kGlobalEtaAddressWidth);
  mismatchGlobalEtaAddress_phiBendLocal = new TH1I("mismatchGlobalEtaAddress_phiBendLocal", "Global Eta Address (Legal Addresses Only) - Phi Bend Local Word (data fields do not match)", 1<<6, 0, 1<<6);
  mismatchGlobalEtaAddress_phiLocal = new TH1I("mismatchGlobalEtaAddress_phiLocal", "Global Eta Address (Legal Addresses Only) - Phi Local Word (data fields do not match)", 1<<2, 0, 1<<2);
  mismatchGlobalEtaAddress_wireGroup = new TH1I("mismatchGlobalEtaAddress_wireGroup", "Global Eta Address (Legal Addresses Only) - Wire Group Word (data fields do not match)", 1<<7, 0, 1<<7);
  mismatchGlobalEtaAddress_cscId = new TH1I("mismatchGlobalEtaAddress_cscId", "Global Eta Address (Legal Addresses Only) - CSC ID Word (data fields do not match)", 1<<4, 0, 1<<4);
  mismatchGlobalPhiMEAddress = new TH1I("mismatchGlobalPhiMEAddress", "Global Phi ME Address (data fields do not match)",
                                        1<<int(CSCBitWidths::kGlobalPhiAddressWidth/2),
                                        0, 1<<CSCBitWidths::kGlobalPhiAddressWidth);
  mismatchGlobalPhiMEAddress_phiLocal = new TH1I("mismatchGlobalPhiMEAddress_phiLocal", "Global Phi ME Address (Legal Addresses Only) - Phi Local Word (data fields do not match)", 1<<10, 0, 1<<10);
  mismatchGlobalPhiMEAddress_wireGroup = new TH1I("mismatchGlobalPhiMEAddress_wireGroup", "Global Phi ME Address (Legal Addresses Only) - Wire Group Word (data fields do not match)", 1<<5, 0, 1<<5);
  mismatchGlobalPhiMEAddress_cscId = new TH1I("mismatchGlobalPhiMEAddress_cscId", "Global Phi ME Address (Legal Addresses Only) - CSC ID Word (data fields do not match)", 1<<4, 0, 1<<4);
  mismatchGlobalPhiMBAddress = new TH1I("mismatchGlobalPhiMBAddress", "Global Phi MB Address (data fields do not match)",
                                        1<<int(CSCBitWidths::kGlobalPhiAddressWidth/2),
                                        0, 1<<CSCBitWidths::kGlobalPhiAddressWidth);
  mismatchGlobalPhiMBAddress_phiLocal = new TH1I("mismatchGlobalPhiMBAddress_phiLocal", "Global Phi MB Address (Legal Addresses Only) - Phi Local Word (data fields do not match)", 1<<10, 0, 1<<10);
  mismatchGlobalPhiMBAddress_wireGroup = new TH1I("mismatchGlobalPhiMBAddress_wireGroup", "Global Phi MB Address  (Legal Addresses Only)- Wire Group Word (data fields do not match)", 1<<5, 0, 1<<5);
  mismatchGlobalPhiMBAddress_cscId = new TH1I("mismatchGlobalPhiMBAddress_cscId", "Global Phi MB Address (Legal Addresses Only) - CSC ID Word (data fields do not match)", 1<<4, 0, 1<<4);
  mismatchPtAddress = new TH1I("mismatchPtAddress", "Pt Address (data fields do not match)",
                               1<<int(CSCBitWidths::kPtAddressWidth/2),
                               0, 1<<CSCBitWidths::kPtAddressWidth);
  mismatchPtAddress_delta12phi = new TH1I("mismatchPtAddress_delta12phi", "Pt Address (Legal Addresses Only) - Delta 12 Phi Word When Only 2 Stations (data fields do not match)", 1<<8, 0, 1<<8);
  mismatchPtAddress_delta23phi = new TH1I("mismatchPtAddress_delta23phi", "Pt Address (Legal Addresses Only) - Delta 23 Phi Word When Only 2 Stations Ot No Track (data fields do not match)", 1<<4, 0, 1<<4);
  mismatchPtAddress_deltaPhi = new TH1I("mismatchPtAddress_deltaPhi", "Pt Address (Legal Addresses Only) - Delta Phi Word When 3 Stations (data fields do not match)", 1<<12, 0, 1<<12);
  mismatchPtAddress_eta = new TH1I("mismatchPtAddress_eta", "Pt Address (Legal Addresses Only) - Eta Word (data fields do not match)", 1<<4, 0, 1<<4);
  mismatchPtAddress_mode = new TH1I("mismatchPtAddress_mode", "Pt Address (Legal Addresses Only) - Mode Word (data fields do not match)", 1<<4, 0, 1<<4);
  mismatchPtAddress_sign = new TH1I("mismatchPtAddress_sign", "Pt Address (Legal Addresses Only) - Sign Word (data fields do not match)", 1<<1, 0, 1<<1);
  
  
  InputVsOutputLocalPhi_1 = new TH2I("InputVsOutputLocalPhi_1", "Data Field vs Address - " + TString(lut1name) + ": Local Phi",
                                     1<<int((CSCBitWidths::kLocalPhiAddressWidth)/2),
                                     0, 1<<CSCBitWidths::kLocalPhiAddressWidth,
                                     //1<<(CSCBitWidths::kLocalPhiDataBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth),
                                     //0, 1<<(CSCBitWidths::kLocalPhiDataBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth) );
                                     1<<CSCBitWidths::kLocalPhiDataBitWidth,
                                     0, 1<<CSCBitWidths::kLocalPhiDataBitWidth );
  InputVsOutputLocalPhi_1->GetXaxis()->SetTitle("Address");
  InputVsOutputLocalPhi_1->GetYaxis()->SetTitle("Data Field");
  
  InputVsOutputGlobalEta_1 = new TH2I("InputVsOutputGlobalEta_1", "Data Field vs Address - " + TString(lut1name) + ": Global Eta",
                                      1<<int((CSCBitWidths::kGlobalEtaAddressWidth)/2),
                                      0, 1<<CSCBitWidths::kGlobalEtaAddressWidth,
                                      //1<<int(CSCBitWidths::kGlobalEtaBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth - 1),
                                      //0, 1<<(CSCBitWidths::kGlobalEtaBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth - 1) ); 
                                      1<<CSCBitWidths::kGlobalEtaBitWidth,
                                      0, 1<<CSCBitWidths::kGlobalEtaBitWidth ); 
  InputVsOutputGlobalEta_1->GetXaxis()->SetTitle("Address");
  InputVsOutputGlobalEta_1->GetYaxis()->SetTitle("Data Field");
  
  InputVsOutputGlobalPhiME_1 = new TH2I("InputVsOutputGlobalPhiME_1", "Data Field vs Address - " + TString(lut1name) + ": Global Phi ME",
                                        1<<int((CSCBitWidths::kGlobalPhiAddressWidth)/2),
                                        0, 1<<CSCBitWidths::kGlobalPhiAddressWidth,
                                        1<<CSCBitWidths::kGlobalPhiDataBitWidth,
                                        0, 1<<CSCBitWidths::kGlobalPhiDataBitWidth );
  InputVsOutputGlobalPhiME_1->GetXaxis()->SetTitle("Address");
  InputVsOutputGlobalPhiME_1->GetYaxis()->SetTitle("Data Field");
  
  InputVsOutputGlobalPhiMB_1 = new TH2I("InputVsOutputGlobalPhiMB_1", "Data Field vs Address - " + TString(lut1name) + ": Global Phi MB",
                                        1<<int((CSCBitWidths::kGlobalPhiAddressWidth)/2),
                                        0, 1<<CSCBitWidths::kGlobalPhiAddressWidth,
                                        1<<CSCBitWidths::kGlobalPhiDataBitWidth,
                                        0, 1<<CSCBitWidths::kGlobalPhiDataBitWidth );
  
  InputVsOutputGlobalPhiMB_1->GetXaxis()->SetTitle("Address");
  InputVsOutputGlobalPhiMB_1->GetYaxis()->SetTitle("Data Field");
  
  InputVsOutputPt_1 = new TH2I("InputVsOutputPt_1", "Data Field vs Address - " + TString(lut1name) + ": Pt",
                               1<<int((CSCBitWidths::kPtAddressWidth)/2),
                               0, 1<<CSCBitWidths::kPtAddressWidth,
                               1<<int(16/2),
                               0, 1<<16 );
  InputVsOutputPt_1->GetXaxis()->SetTitle("Address");
  InputVsOutputPt_1->GetYaxis()->SetTitle("Data Field");
  
  InputVsOutputLocalPhi_2 = new TH2I("InputVsOutputLocalPhi_2", "Data Field vs Address - " + TString(lut2name) + ": Local Phi",
                                     1<<int((CSCBitWidths::kLocalPhiAddressWidth)/2),
                                     0, 1<<CSCBitWidths::kLocalPhiAddressWidth,
                                     //1<<(CSCBitWidths::kLocalPhiDataBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth),
                                     //0, 1<<(CSCBitWidths::kLocalPhiDataBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth) );
                                     1<<CSCBitWidths::kLocalPhiDataBitWidth,
                                     0, 1<<CSCBitWidths::kLocalPhiDataBitWidth );
  InputVsOutputLocalPhi_2->GetXaxis()->SetTitle("Address");
  InputVsOutputLocalPhi_2->GetYaxis()->SetTitle("Data Field");
  
  InputVsOutputGlobalEta_2 = new TH2I("InputVsOutputGlobalEta_2", "Data Field vs Address - " + TString(lut2name) + ": Global Eta",
                                      1<<int((CSCBitWidths::kGlobalEtaAddressWidth)/2),
                                      0, 1<<CSCBitWidths::kGlobalEtaAddressWidth,
                                      //1<<(CSCBitWidths::kGlobalEtaBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth - 1),
                                      //0, 1<<(CSCBitWidths::kGlobalEtaBitWidth + CSCBitWidths::kLocalPhiBendDataBitWidth - 1) ); 
                                      1<<CSCBitWidths::kGlobalEtaBitWidth,
                                      0, 1<<CSCBitWidths::kGlobalEtaBitWidth ); 
  InputVsOutputGlobalEta_2->GetXaxis()->SetTitle("Address");
  InputVsOutputGlobalEta_2->GetYaxis()->SetTitle("Data Field");
  
  InputVsOutputGlobalPhiME_2 = new TH2I("InputVsOutputGlobalPhiME_2", "Data Field vs Address - " + TString(lut2name) + ": Global Phi ME",
                                        1<<int((CSCBitWidths::kGlobalPhiAddressWidth)/2),
                                        0, 1<<CSCBitWidths::kGlobalPhiAddressWidth,
                                        1<<CSCBitWidths::kGlobalPhiDataBitWidth,
                                        0, 1<<CSCBitWidths::kGlobalPhiDataBitWidth );
  InputVsOutputGlobalPhiME_2->GetXaxis()->SetTitle("Address");
  InputVsOutputGlobalPhiME_2->GetYaxis()->SetTitle("Data Field");
  
  InputVsOutputGlobalPhiMB_2 = new TH2I("InputVsOutputGlobalPhiMB_2", "Data Field vs Address - " + TString(lut2name) + ": Global Phi MB",
                                        1<<int((CSCBitWidths::kGlobalPhiAddressWidth)/2),
                                        0, 1<<CSCBitWidths::kGlobalPhiAddressWidth,
                                        1<<CSCBitWidths::kGlobalPhiDataBitWidth,
                                        0, 1<<CSCBitWidths::kGlobalPhiDataBitWidth );
  InputVsOutputGlobalPhiMB_2->GetXaxis()->SetTitle("Address");
  InputVsOutputGlobalPhiMB_2->GetYaxis()->SetTitle("Data Field");
  
  InputVsOutputPt_2 = new TH2I("InputVsOutputPt_2", "Data Field vs Address - " + TString(lut2name) + ": Pt",
                               1<<int((CSCBitWidths::kPtAddressWidth)/2),
                               0, 1<<CSCBitWidths::kPtAddressWidth,
                               1<<int(16/2),
                               0, 1<<16 );
  InputVsOutputPt_2->GetXaxis()->SetTitle("Address");
  InputVsOutputPt_2->GetYaxis()->SetTitle("Data Field");
}

CSCCompareLUTs::~CSCCompareLUTs()
{
  delete compareLocalPhi;
  compareLocalPhi = NULL;
  delete compareLocalPhi_phiLocal;
  compareLocalPhi_phiLocal = NULL;
  delete compareLocalPhi_phiBend;
  compareLocalPhi_phiBend = NULL;
  delete compareGlobalEta;
  compareGlobalEta = NULL;
  delete compareGlobalEta_etaGlobal;
  compareGlobalEta_etaGlobal = NULL;
  delete compareGlobalEta_phiBend;
  compareGlobalEta_phiBend = NULL;
  delete compareGlobalPhiME;
  compareGlobalPhiME = NULL;
  delete compareGlobalPhiMB;
  compareGlobalPhiMB = NULL;
  delete comparePt;
  comparePt = NULL;
  delete comparePt_front;
  comparePt_front = NULL;
  delete comparePt_rear;
  comparePt_rear = NULL;
  
  delete compareLocalPhiOffDiagonal;
  compareLocalPhiOffDiagonal = NULL;
  delete compareLocalPhiOffDiagonal_phiLocal;
  compareLocalPhiOffDiagonal_phiLocal = NULL;
  delete compareLocalPhiOffDiagonal_phiBend;
  compareLocalPhiOffDiagonal_phiBend = NULL;
  delete compareGlobalEtaOffDiagonal;
  compareGlobalEtaOffDiagonal = NULL;
  delete compareGlobalEtaOffDiagonal_etaGlobal;
  compareGlobalEtaOffDiagonal_etaGlobal = NULL;
  delete compareGlobalEtaOffDiagonal_phiBend;
  compareGlobalEtaOffDiagonal_phiBend = NULL;
  delete compareGlobalPhiMEOffDiagonal;
  compareGlobalPhiMEOffDiagonal = NULL;
  delete compareGlobalPhiMBOffDiagonal;
  compareGlobalPhiMBOffDiagonal = NULL;
  delete comparePtOffDiagonal;
  comparePtOffDiagonal = NULL;
  delete comparePtOffDiagonal_front;
  comparePtOffDiagonal_front = NULL;
  delete comparePtOffDiagonal_rear;
  comparePtOffDiagonal_rear = NULL;
  
  delete differenceLocalPhi;
  differenceLocalPhi = NULL;
  delete differenceLocalPhi_phiLocal;
  differenceLocalPhi_phiLocal = NULL;
  delete differenceLocalPhi_phiBend;
  differenceLocalPhi_phiBend = NULL;
  delete differenceGlobalEta;
  differenceGlobalEta = NULL;
  delete differenceGlobalEta_etaGlobal;
  differenceGlobalEta_etaGlobal = NULL;
  delete differenceGlobalEta_phiBend;
  differenceGlobalEta_phiBend = NULL;
  delete differenceGlobalPhiME;
  differenceGlobalPhiME = NULL;
  delete differenceGlobalPhiMB;
  differenceGlobalPhiMB = NULL;
  delete differencePt;
  differencePt = NULL;
  delete differencePt_front;
  differencePt_front = NULL;
  delete differencePt_rear;
  differencePt_rear = NULL;
  
  delete mismatchLocalPhiAddress;
  mismatchLocalPhiAddress = NULL;
  delete mismatchLocalPhiAddress_patternId;
  mismatchLocalPhiAddress_patternId = NULL;
  delete mismatchLocalPhiAddress_patternNumber;
  mismatchLocalPhiAddress_patternNumber = NULL;
  delete mismatchLocalPhiAddress_quality;
  mismatchLocalPhiAddress_quality = NULL;
  delete mismatchLocalPhiAddress_leftRight;
  mismatchLocalPhiAddress_leftRight = NULL;
  delete mismatchLocalPhiAddress_spare;
  mismatchLocalPhiAddress_spare = NULL;
  delete mismatchGlobalEtaAddress;
  mismatchGlobalEtaAddress = NULL;
  delete mismatchGlobalEtaAddress_phiBendLocal;
  mismatchGlobalEtaAddress_phiBendLocal = NULL;
  delete mismatchGlobalEtaAddress_phiLocal;
  mismatchGlobalEtaAddress_phiLocal = NULL;
  delete mismatchGlobalEtaAddress_wireGroup;
  mismatchGlobalEtaAddress_wireGroup = NULL;
  delete mismatchGlobalEtaAddress_cscId;
  mismatchGlobalEtaAddress_cscId = NULL;
  delete mismatchGlobalPhiMEAddress;
  mismatchGlobalPhiMEAddress = NULL;
  delete mismatchGlobalPhiMEAddress_phiLocal;
  mismatchGlobalPhiMEAddress_phiLocal = NULL;
  delete mismatchGlobalPhiMEAddress_wireGroup;
  mismatchGlobalPhiMEAddress_wireGroup = NULL;
  delete mismatchGlobalPhiMEAddress_cscId;
  mismatchGlobalPhiMEAddress_cscId = NULL;
  delete mismatchGlobalPhiMBAddress;
  mismatchGlobalPhiMBAddress = NULL;
  delete mismatchGlobalPhiMBAddress_phiLocal;
  mismatchGlobalPhiMBAddress_phiLocal = NULL;
  delete mismatchGlobalPhiMBAddress_wireGroup;
  mismatchGlobalPhiMBAddress_wireGroup = NULL;
  delete mismatchGlobalPhiMBAddress_cscId;
  mismatchGlobalPhiMBAddress_cscId = NULL;
  delete mismatchPtAddress;
  mismatchPtAddress = NULL;
  delete mismatchPtAddress_delta12phi;
  mismatchPtAddress_delta12phi = NULL;
  delete mismatchPtAddress_delta23phi;
  mismatchPtAddress_delta23phi = NULL;
  delete mismatchPtAddress_deltaPhi;
  mismatchPtAddress_deltaPhi = NULL;
  delete mismatchPtAddress_eta;
  mismatchPtAddress_eta = NULL;
  delete mismatchPtAddress_mode;
  mismatchPtAddress_mode = NULL;
  delete mismatchPtAddress_sign;
  mismatchPtAddress_sign = NULL;

  delete InputVsOutputLocalPhi_1;
  InputVsOutputLocalPhi_1 = NULL;
  delete InputVsOutputGlobalEta_1;
  InputVsOutputGlobalEta_1 = NULL;
  delete InputVsOutputGlobalPhiME_1;
  InputVsOutputGlobalPhiME_1 = NULL;
  delete InputVsOutputGlobalPhiMB_1;
  InputVsOutputGlobalPhiMB_1 = NULL;
  delete InputVsOutputPt_1;
  InputVsOutputPt_1 = NULL;
  delete InputVsOutputLocalPhi_2;
  InputVsOutputLocalPhi_2 = NULL;
  delete InputVsOutputGlobalEta_2;
  InputVsOutputGlobalEta_2 = NULL;
  delete InputVsOutputGlobalPhiME_2;
  InputVsOutputGlobalPhiME_2 = NULL;
  delete InputVsOutputGlobalPhiMB_2;
  InputVsOutputGlobalPhiMB_2 = NULL;
  delete InputVsOutputPt_2;
  InputVsOutputPt_2 = NULL;

  delete fCompare;
  fCompare = NULL;
}

void CSCCompareLUTs::analyze(edm::Event const& e, edm::EventSetup const& iSetup)
{
  //  compareLUTs(SRLUT_Base_1, SRLUT_Base_2, PtLUT_Base_1, PtLUT_Base_2, true, true, true, true, true);

  // set geometry pointer
  edm::ESHandle<CSCGeometry> pDD;

  iSetup.get<MuonGeometryRecord>().get( pDD );
  CSCTriggerGeometry::setGeometry(pDD);

  edm::ESHandle< L1MuTriggerScales > scales ;
  iSetup.get< L1MuTriggerScalesRcd >().get( scales ) ;

  edm::ESHandle< L1MuTriggerPtScale > ptScale ;
  iSetup.get< L1MuTriggerPtScaleRcd >().get( ptScale ) ;

  // Storing flags to be passed to the SR LUT objects.  This is done because when reading a file,
  // the file name for each LUT must be passed to the SR LUT object.  This is a way to read all the
  // LUT files within a given release by running this program once
  std::string LUTPath_1 = lutParam1.getUntrackedParameter<std::string>("LUTPath", std::string("L1Trigger/CSCTrackFinder/LUTs"));
  std::string LUTPath_2 = lutParam2.getUntrackedParameter<std::string>("LUTPath", std::string("L1Trigger/CSCTrackFinder/LUTs"));
  
  bool readSRLUTs_1 = lutParam1.getUntrackedParameter<bool>("ReadLUTs",false);
  bool readSRLUTs_2 = lutParam2.getUntrackedParameter<bool>("ReadLUTs",false);
  bool readPtLUTs_1 = lutParam1.getUntrackedParameter<bool>("ReadPtLUT",false);
  bool readPtLUTs_2 = lutParam2.getUntrackedParameter<bool>("ReadPtLUT",false);
  bool isSRBinary_1 = lutParam1.getUntrackedParameter<bool>("Binary",false);
  bool isSRBinary_2 = lutParam2.getUntrackedParameter<bool>("Binary",false);
  bool isPtBinary_1 = lutParam1.getUntrackedParameter<bool>("isBinary",false);
  bool isPtBinary_2 = lutParam2.getUntrackedParameter<bool>("isBinary",false);
  bool isTMB07_1 = lutParam1.getUntrackedParameter<bool>("isTMB07",false);
  bool isTMB07_2 = lutParam2.getUntrackedParameter<bool>("isTMB07",false);
  bool useMiniLUTs_1 = lutParam1.getUntrackedParameter<bool>("UseMiniLUTs",false);
  bool useMiniLUTs_2 = lutParam2.getUntrackedParameter<bool>("UseMiniLUTs",false);

  edm::FileInPath me_lcl_phi_file_1;
  edm::FileInPath me_lcl_phi_file_2;
  edm::FileInPath me_gbl_phi_file_1;
  edm::FileInPath me_gbl_phi_file_2;
  edm::FileInPath mb_gbl_phi_file_1;
  edm::FileInPath mb_gbl_phi_file_2;
  edm::FileInPath me_gbl_eta_file_1;
  edm::FileInPath me_gbl_eta_file_2;
  edm::FileInPath pt_lut_file_1;  
  edm::FileInPath pt_lut_file_2;  
  
  if(readSRLUTs_1)
    me_lcl_phi_file_1 = edm::FileInPath(LUTPath_1 + std::string("/LocalPhiLUT") + (isSRBinary_1 ? std::string(".bin") : std::string(".dat")));
  if(readSRLUTs_2)
    me_lcl_phi_file_2 = edm::FileInPath(LUTPath_2 + std::string("/LocalPhiLUT") + (isSRBinary_2 ? std::string(".bin") : std::string(".dat")));
  if(readPtLUTs_1)
    pt_lut_file_1 = edm::FileInPath(LUTPath_1 + std::string("/L1CSCPtLUT") + (isPtBinary_1 ? std::string(".bin") : std::string(".dat")));
  if(readPtLUTs_2)
    pt_lut_file_2 = edm::FileInPath(LUTPath_2 + std::string("/L1CSCPtLUT") + (isPtBinary_2 ? std::string(".bin") : std::string(".dat")));
  
  for(int endcapItr=1;endcapItr<=2;endcapItr++)
    if(endcapItr==endcap || endcap==-1)
      for(int sectorItr=1;sectorItr<=6;sectorItr++)
	if(sectorItr==sector || sector==-1)
	  for(int stationItr=1;stationItr<=4;stationItr++)
	    if(stationItr==station || station==-1)
	      {
		if(stationItr==1)
		  {
		    for(int subsectorItr=1;subsectorItr<=2;subsectorItr++)
		      {
			if(subsectorItr==subsector || subsector==-1)
			  {
			    if(readSRLUTs_1)
			      {
				me_gbl_phi_file_1 = edm::FileInPath(LUTPath_1 + std::string("/GlobalPhiME") + 
								    encodeFileIndex(endcapItr,sectorItr,stationItr,subsectorItr) + 
								    (isSRBinary_1 ? std::string(".bin") : std::string(".dat")));
				mb_gbl_phi_file_1 = edm::FileInPath(LUTPath_1 + std::string("/GlobalPhiMB") + 
								    encodeFileIndex(endcapItr,sectorItr,stationItr,subsectorItr) + 
								    (isSRBinary_1 ? std::string(".bin") : std::string(".dat")));
				me_gbl_eta_file_1 = edm::FileInPath(LUTPath_1 + std::string("/GlobalEtaME") + 
								    encodeFileIndex(endcapItr,sectorItr,stationItr,subsectorItr) + 
								    (isSRBinary_1 ? std::string(".bin") : std::string(".dat")));
			      }
			    if(readSRLUTs_2)
			      {
				me_gbl_phi_file_2 = edm::FileInPath(LUTPath_2 + std::string("/GlobalPhiME") + 
								    encodeFileIndex(endcapItr,sectorItr,stationItr,subsectorItr) + 
								    (isSRBinary_2 ? std::string(".bin") : std::string(".dat")));
				mb_gbl_phi_file_2 = edm::FileInPath(LUTPath_2 + std::string("/GlobalPhiMB") + 
								    encodeFileIndex(endcapItr,sectorItr,stationItr,subsectorItr) + 
								    (isSRBinary_2 ? std::string(".bin") : std::string(".dat")));
				me_gbl_eta_file_2 = edm::FileInPath(LUTPath_2 + std::string("/GlobalEtaME") + 
								    encodeFileIndex(endcapItr,sectorItr,stationItr,subsectorItr) + 
								    (isSRBinary_2 ? std::string(".bin") : std::string(".dat")));
			      }
			    
			    edm::ParameterSet _lutParam_1;
			    edm::ParameterSet _lutParam_2;
			    
			    _lutParam_1.addUntrackedParameter("ReadLUTs", readSRLUTs_1);
			    _lutParam_2.addUntrackedParameter("ReadLUTs", readSRLUTs_2);
			    _lutParam_1.addUntrackedParameter("ReadPtLUT", false);
			    _lutParam_2.addUntrackedParameter("ReadPtLUT", false);
			    //_lutParam_1.addUntrackedParameter("isTMB07", isTMB07_1);
			    //_lutParam_2.addUntrackedParameter("isTMB07", isTMB07_2);
			    _lutParam_1.addUntrackedParameter("UseMiniLUTs", useMiniLUTs_1);
			    _lutParam_2.addUntrackedParameter("UseMiniLUTs", useMiniLUTs_2);

			    if(readSRLUTs_1)
			      {
				_lutParam_1.addUntrackedParameter<bool>("Binary", isSRBinary_1);
				_lutParam_1.addUntrackedParameter("LocalPhiLUT", me_lcl_phi_file_1);
				_lutParam_1.addUntrackedParameter("GlobalPhiLUTME", me_gbl_phi_file_1);
				_lutParam_1.addUntrackedParameter("GlobalPhiLUTMB", mb_gbl_phi_file_1);
				_lutParam_1.addUntrackedParameter("GlobalEtaLUTME", me_gbl_eta_file_1);
			      }
			    if(readPtLUTs_1)
			      {
				_lutParam_1.addUntrackedParameter("isBinary", isPtBinary_1);
				_lutParam_1.addUntrackedParameter("PtLUTFile", pt_lut_file_1);
			      }
			    if(readSRLUTs_2)
			      {
				_lutParam_2.addUntrackedParameter<bool>("Binary", isSRBinary_2);
				_lutParam_2.addUntrackedParameter("LocalPhiLUT", me_lcl_phi_file_2);
				_lutParam_2.addUntrackedParameter("GlobalPhiLUTME", me_gbl_phi_file_2);
				_lutParam_2.addUntrackedParameter("GlobalPhiLUTMB", mb_gbl_phi_file_2);
				_lutParam_2.addUntrackedParameter("GlobalEtaLUTME", me_gbl_eta_file_2);
			      }
			    if(readPtLUTs_2)
			      {
				_lutParam_2.addUntrackedParameter("isBinary", isPtBinary_2);
				_lutParam_2.addUntrackedParameter("PtLUTFile", pt_lut_file_2);
			      }
			    
			    CSCSectorReceiverLUT SRLUT_1(endcapItr, sectorItr, subsectorItr, stationItr, _lutParam_1, isTMB07_1);
			    CSCSectorReceiverLUT SRLUT_2(endcapItr, sectorItr, subsectorItr, stationItr, _lutParam_2, isTMB07_2);
			    CSCTFPtLUT PtLUT_1(_lutParam_1, scales.product(), ptScale.product());
			    CSCTFPtLUT PtLUT_2(_lutParam_2, scales.product(), ptScale.product());
			    
			    compareLUTs(&SRLUT_1, &SRLUT_2, &PtLUT_1, &PtLUT_2, false, true, true, true, false, endcapItr, sectorItr, subsectorItr, stationItr);
			  }
		      }
		  }
		else
		  {
		    if(readSRLUTs_1)
		      {
			me_gbl_phi_file_1 = edm::FileInPath(LUTPath_1 + std::string("/GlobalPhiME") + 
							    encodeFileIndex(endcapItr,sectorItr,stationItr) + 
							    (isSRBinary_1 ? std::string(".bin") : std::string(".dat")));
			mb_gbl_phi_file_1 = edm::FileInPath(LUTPath_1 + std::string("/GlobalPhiMB") + 
							    encodeFileIndex(endcapItr,sectorItr,1,1) + 
							    (isSRBinary_1 ? std::string(".bin") : std::string(".dat")));
			me_gbl_eta_file_1 = edm::FileInPath(LUTPath_1 + std::string("/GlobalEtaME") + 
							    encodeFileIndex(endcapItr,sectorItr,stationItr) + 
							    (isSRBinary_1 ? std::string(".bin") : std::string(".dat")));
		      }
		    if(readSRLUTs_2)
		      {
			me_gbl_phi_file_2 = edm::FileInPath(LUTPath_2 + std::string("/GlobalPhiME") + 
							    encodeFileIndex(endcapItr,sectorItr,stationItr) + 
							    (isSRBinary_2 ? std::string(".bin") : std::string(".dat")));
			mb_gbl_phi_file_2 = edm::FileInPath(LUTPath_2 + std::string("/GlobalPhiMB") + 
							    encodeFileIndex(endcapItr,sectorItr,1,1) + 
							    (isSRBinary_2 ? std::string(".bin") : std::string(".dat")));
			me_gbl_eta_file_2 = edm::FileInPath(LUTPath_2 + std::string("/GlobalEtaME") + 
							    encodeFileIndex(endcapItr,sectorItr,stationItr) + 
							    (isSRBinary_2 ? std::string(".bin") : std::string(".dat")));
		      }
		    
		    edm::ParameterSet _lutParam_1;
		    edm::ParameterSet _lutParam_2;
		    
		    _lutParam_1.addUntrackedParameter("ReadLUTs", readSRLUTs_1);
		    _lutParam_2.addUntrackedParameter("ReadLUTs", readSRLUTs_2);
		    _lutParam_1.addUntrackedParameter("ReadPtLUT", false);
		    _lutParam_2.addUntrackedParameter("ReadPtLUT", false);
		    //_lutParam_1.addUntrackedParameter("isTMB07", isTMB07_1);
		    //_lutParam_2.addUntrackedParameter("isTMB07", isTMB07_2);
            _lutParam_1.addUntrackedParameter("UseMiniLUTs", useMiniLUTs_1);
            _lutParam_2.addUntrackedParameter("UseMiniLUTs", useMiniLUTs_2);
		    
		    if(readSRLUTs_1)
		      {
			_lutParam_1.addUntrackedParameter("Binary", isSRBinary_1);
			_lutParam_1.addUntrackedParameter("LocalPhiLUT", me_lcl_phi_file_1);
			_lutParam_1.addUntrackedParameter("GlobalPhiLUTME", me_gbl_phi_file_1);
			_lutParam_1.addUntrackedParameter("GlobalPhiLUTMB", mb_gbl_phi_file_1);
			_lutParam_1.addUntrackedParameter("GlobalEtaLUTME", me_gbl_eta_file_1);
		      }
		    if(readPtLUTs_1)
		      {
			_lutParam_1.addUntrackedParameter("isBinary", isPtBinary_1);
			_lutParam_1.addUntrackedParameter("PtLUTFile", pt_lut_file_1);
		      }
		    if(readSRLUTs_2)
		      {
			_lutParam_2.addUntrackedParameter("Binary", isSRBinary_2);
			_lutParam_2.addUntrackedParameter("LocalPhiLUT", me_lcl_phi_file_2);
			_lutParam_2.addUntrackedParameter("GlobalPhiLUTME", me_gbl_phi_file_2);
			_lutParam_2.addUntrackedParameter("GlobalPhiLUTMB", mb_gbl_phi_file_2);
			_lutParam_2.addUntrackedParameter("GlobalEtaLUTME", me_gbl_eta_file_2);
		      }
		    if(readPtLUTs_2)
		      {
			_lutParam_2.addUntrackedParameter("isBinary", isPtBinary_2);
			_lutParam_2.addUntrackedParameter("PtLUTFile", pt_lut_file_2);
		      }
		    
		    CSCSectorReceiverLUT SRLUT_1(endcapItr, sectorItr, 1, stationItr, _lutParam_1, isTMB07_1);
		    CSCSectorReceiverLUT SRLUT_2(endcapItr, sectorItr, 1, stationItr, _lutParam_2, isTMB07_2);
		    CSCTFPtLUT PtLUT_1(_lutParam_1, scales.product(), ptScale.product());
		    CSCTFPtLUT PtLUT_2(_lutParam_2, scales.product(), ptScale.product());

		    compareLUTs(&SRLUT_1, &SRLUT_2, &PtLUT_1, &PtLUT_2, false, true, true, false, false, endcapItr, sectorItr, 1, stationItr);
		  }
	      }
  
  edm::ParameterSet _lutParam_1;
  edm::ParameterSet _lutParam_2;
  
  _lutParam_1.addUntrackedParameter<bool>("ReadLUTs", readSRLUTs_1);
  _lutParam_2.addUntrackedParameter<bool>("ReadLUTs", readSRLUTs_2);
  _lutParam_1.addUntrackedParameter<bool>("ReadPtLUT", readPtLUTs_1);
  _lutParam_2.addUntrackedParameter<bool>("ReadPtLUT", readPtLUTs_2);
  _lutParam_1.addUntrackedParameter("isTMB07", isTMB07_1);
  _lutParam_2.addUntrackedParameter("isTMB07", isTMB07_2);
  _lutParam_1.addUntrackedParameter("UseMiniLUTs", useMiniLUTs_1);
  _lutParam_2.addUntrackedParameter("UseMiniLUTs", useMiniLUTs_2);
  
  if(readSRLUTs_1)
    {
      _lutParam_1.addUntrackedParameter<bool>("Binary", isSRBinary_1);
      _lutParam_1.addUntrackedParameter<edm::FileInPath>("LocalPhiLUT", me_lcl_phi_file_1);
      _lutParam_1.addUntrackedParameter("GlobalPhiLUTME", me_gbl_phi_file_1);
      _lutParam_1.addUntrackedParameter("GlobalPhiLUTMB", mb_gbl_phi_file_1);
      _lutParam_1.addUntrackedParameter("GlobalEtaLUTME", me_gbl_eta_file_1);
    }
  if(readPtLUTs_1)
    {
      _lutParam_1.addUntrackedParameter<bool>("isBinary", isPtBinary_1);
      _lutParam_1.addUntrackedParameter<edm::FileInPath>("PtLUTFile", pt_lut_file_1);
    }
  if(readSRLUTs_2)
    {
      _lutParam_2.addUntrackedParameter<bool>("Binary", isSRBinary_2);
      _lutParam_2.addUntrackedParameter<edm::FileInPath>("LocalPhiLUT", me_lcl_phi_file_2);
      _lutParam_2.addUntrackedParameter("GlobalPhiLUTME", me_gbl_phi_file_2);
      _lutParam_2.addUntrackedParameter("GlobalPhiLUTMB", mb_gbl_phi_file_2);
      _lutParam_2.addUntrackedParameter("GlobalEtaLUTME", me_gbl_eta_file_2);
    }
  if(readPtLUTs_2)
    {
      _lutParam_2.addUntrackedParameter<bool>("isBinary", isPtBinary_2);
      _lutParam_2.addUntrackedParameter<edm::FileInPath>("PtLUTFile", pt_lut_file_2);
    }
  
  CSCSectorReceiverLUT SRLUT_1(1, 1, 1, 1, _lutParam_1,isTMB07_1);
  CSCSectorReceiverLUT SRLUT_2(1, 1, 1, 1, _lutParam_2,isTMB07_2);
  CSCTFPtLUT PtLUT_1(_lutParam_1, scales.product(), ptScale.product());
  CSCTFPtLUT PtLUT_2(_lutParam_2, scales.product(), ptScale.product());
  
  compareLUTs(&SRLUT_1, &SRLUT_2, &PtLUT_1, &PtLUT_2, true, false, false, false, true, 1, 1, 1, 1);
}


void CSCCompareLUTs::endJob()
{
  fCompare->Write();
}


void CSCCompareLUTs::compareLUTs(CSCSectorReceiverLUT *SRLUT_1, CSCSectorReceiverLUT *SRLUT_2, CSCTFPtLUT *PtLUT_1, CSCTFPtLUT *PtLUT_2,
                                 bool doLocalPhi, bool doGlobalEta, bool doGlobalPhiME, bool doGlobalPhiMB, bool doPt,
                                 const int _endcap, const int _sector, const int _subsector, const int _station)
{
  // test local phi
  if(doLocalPhi)
    {
      for(unsigned int address = 0; address < 1<<CSCBitWidths::kLocalPhiAddressWidth; ++address)
        {
          lclphidat out_1, out_2;
          unsigned short LUT_1, LUT_2;
          bool legalAddress = true;
          
          try
          {
            out_1 = SRLUT_1->localPhi(address);
            LUT_1 = out_1.toint();
          }
          catch(...)
          {
            LUT_1 = 0;
          }
          try
          {
            out_2 = SRLUT_2->localPhi(address);
            LUT_2 = out_2.toint();
          }
          catch(...)
          {
            LUT_2 = 0;
          }
          
          if((address & 0xFF) >= 2*CSCConstants::MAX_NUM_STRIPS          || // strip out of bounds
             out_1.phi_local >= (1<<CSCBitWidths::kLocalPhiDataBitWidth) || // phiLocal word of out_1 data field out of bounds
             out_2.phi_local >= (1<<CSCBitWidths::kLocalPhiDataBitWidth) )  // phiLocal word of out_2 data field out of bounds
            {
              legalAddress=false;
            }
          
          compareLocalPhi->Fill(LUT_1, LUT_2);
          compareLocalPhi_phiLocal->Fill((LUT_1 & 0x3ff), (LUT_2 & 0x3ff));
          compareLocalPhi_phiBend->Fill(((LUT_1 & 0xfc00)>>10), ((LUT_2 & 0xfc00)>>10));
          
          if(LUT_1 != LUT_2)
            {
              mismatchLocalPhiAddress->Fill(address);
              if(legalAddress)
                {
                  mismatchLocalPhiAddress_patternId->Fill(address & 0xFF);
                  mismatchLocalPhiAddress_patternNumber->Fill((address & 0xf00) >> 8);
                  mismatchLocalPhiAddress_quality->Fill((address & 0xf000) >> 12);
                  mismatchLocalPhiAddress_leftRight->Fill((address & 0x10000) >> 16);
                  mismatchLocalPhiAddress_spare->Fill((address & 0x60000) >> 17);
                }
              
              compareLocalPhiOffDiagonal->Fill(LUT_1,LUT_2);
            } 

          differenceLocalPhi->Fill(LUT_1 - LUT_2);
          differenceLocalPhi_phiLocal->Fill((LUT_1 & 0x3ff) - (LUT_2 & 0x3ff));
          differenceLocalPhi_phiBend->Fill(((LUT_1 & 0xfc00)>>10) - ((LUT_2 & 0xfc00)>>10));
          
          if((LUT_1 & 0x3ff) != (LUT_2 & 0x3ff)) // if phi local word does not match
            {
              compareLocalPhiOffDiagonal_phiLocal->Fill((LUT_1 & 0x3ff), (LUT_2 & 0x3ff));
            }
          
          if((LUT_1 & 0xfc00) != (LUT_2 & 0xfc00)) // if phi bend word does not match
            {
              compareLocalPhiOffDiagonal_phiBend->Fill((LUT_1 & 0xfc00)>>10, (LUT_2 & 0xfc00)>>10);
            }
          
          InputVsOutputLocalPhi_1->Fill(address, LUT_1);
          InputVsOutputLocalPhi_2->Fill(address, LUT_2);
        }
    }
  
  //test global phi ME
  if(doGlobalPhiME)
    {
      for(unsigned int address = 0; address < 1<<CSCBitWidths::kGlobalPhiAddressWidth; ++address)
        {
          gblphidat out_1, out_2; 
          unsigned short LUT_1, LUT_2;
          bool legalAddress = true;
          
          // compare global phi ME
          try
          {
            out_1 = SRLUT_1->globalPhiME(address);
            LUT_1 = out_1.toint();
          }
          catch(...)
          {
            LUT_1 = 0;
          }
          try
          {
            out_2 = SRLUT_2->globalPhiME(address);
            LUT_2 = out_2.toint();
          }
          catch(...)
          {
            LUT_2 = 0;
          }
          
          
          if(((address & 0x78000) >> 15) < CSCTriggerNumbering::minTriggerCscId() || // CSC ID out of bounds
             ((address & 0x78000) >> 15) > CSCTriggerNumbering::maxTriggerCscId() || // CSC ID out of bounds
             ((address & 0x7c00) >> 10) >= 1<<5                                   || // Wire group out of bounds
             (address & 0x3ff) >= 1<<CSCBitWidths::kLocalPhiDataBitWidth)            // phi local word it out of bounds
    	    {
              legalAddress=false;
            }
          
          
          
          compareGlobalPhiME->Fill(LUT_1, LUT_2);
          
          if(LUT_1 != LUT_2)
            {
              mismatchGlobalPhiMEAddress->Fill(address);
              if(legalAddress)
                {
                  mismatchGlobalPhiMEAddress_phiLocal->Fill(address & 0x3ff);
                  mismatchGlobalPhiMEAddress_wireGroup->Fill((address & 0x7c00) >> 10);
                  mismatchGlobalPhiMEAddress_cscId->Fill((address & 0x78000) >> 15);
                }
              
              compareGlobalPhiMEOffDiagonal->Fill(LUT_1, LUT_2);
            }
          
          differenceGlobalPhiME->Fill(LUT_1 - LUT_2);
          
          InputVsOutputGlobalPhiME_1->Fill(address,LUT_1);
          InputVsOutputGlobalPhiME_2->Fill(address,LUT_2);
        }
    }
  
  //test global phi MB
  if(doGlobalPhiMB)
    {
      for(unsigned int address = 0; address < 1<<CSCBitWidths::kGlobalPhiAddressWidth; ++address)
        {
          gblphidat out_1, out_2; 
          unsigned short LUT_1, LUT_2;
          bool legalAddress = true;
          
          try
          {
            out_1 = SRLUT_1->globalPhiMB(address);
            LUT_1 = out_1.toint();
          }
          catch(...)
          {
            LUT_1 = 0;
          }
          try
          {
            out_2 = SRLUT_2->globalPhiMB(address);
            LUT_2 = out_2.toint();
          }
          catch(...)
          {
            LUT_2 = 0;
          }
          
          if(((address & 0x78000) >> 15) < CSCTriggerNumbering::minTriggerCscId() || // CSC ID out of bounds
             ((address & 0x78000) >> 15) > CSCTriggerNumbering::maxTriggerCscId() || // CSC ID out of bounds
             ((address & 0x7c00) >> 10) > 128                                     || // wire group out of bounds
             (address & 0x3ff) >= 1<<CSCBitWidths::kLocalPhiDataBitWidth)            // phi local word it out of bounds
            {
              legalAddress=false;
            }
          
          compareGlobalPhiMB->Fill(LUT_1, LUT_2);
          
          if(LUT_1 != LUT_2)
            {
              mismatchGlobalPhiMBAddress->Fill(address);
              if(legalAddress)
                {
                  mismatchGlobalPhiMBAddress_phiLocal->Fill(address & 0x3ff);
                  mismatchGlobalPhiMBAddress_wireGroup->Fill((address & 0x7c00) >> 10);
                  mismatchGlobalPhiMBAddress_cscId->Fill((address & 0x78000) >> 15);
                }
              
              compareGlobalPhiMBOffDiagonal->Fill(LUT_1, LUT_2);
            }
              
          differenceGlobalPhiMB->Fill(LUT_1 - LUT_2);
          
          InputVsOutputGlobalPhiMB_1->Fill(address,LUT_1);
          InputVsOutputGlobalPhiMB_2->Fill(address,LUT_2);      
        }
    }
  
  //test global eta
  if(doGlobalEta)
    {
      for(unsigned int address = 0; address < 1<<CSCBitWidths::kGlobalEtaAddressWidth; ++address)
        {
          gbletadat out_1, out_2;
          unsigned short LUT_1, LUT_2;
          bool legalAddress = true;
          
          try
          {
            out_1 = SRLUT_1->globalEtaME(address);
            LUT_1 = out_1.toint();
          }
          catch(...)
          {
            LUT_1 = 0;
          }
          try
          {
            out_2 = SRLUT_2->globalEtaME(address);
            LUT_2 = out_2.toint();
          }
          catch(...)
          {
            LUT_2 = 0;
          }
          
          double float_eta = getGlobalEtaValue(_endcap, _sector, _subsector, station,
                                               ((address >>15) & 0xf), ((address >> 8) & 0x7f), ((address >> 6) & 0x3));
          if ((float_eta < CSCTFConstants::minEta) || (float_eta >= CSCTFConstants::maxEta) || // eta out of range
              ((address >> 15) & 0xf) < CSCTriggerNumbering::minTriggerCscId()              || // CSC ID out of range
              ((address >> 15) & 0xf) > CSCTriggerNumbering::maxTriggerCscId() )               // CSC ID out of range
            {
              legalAddress=false;
            }
          
          compareGlobalEta->Fill(LUT_1, LUT_2);
          compareGlobalEta_etaGlobal->Fill((LUT_1 & 0x7f), (LUT_2 & 0x7f));
          compareGlobalEta_phiBend->Fill(((LUT_1 >> 7) & 0x1f), ((LUT_2 >> 7) & 0x1f));
          
          if(LUT_1 != LUT_2)
            {
              mismatchGlobalEtaAddress->Fill(address);
              if(legalAddress)
                {
                  mismatchGlobalEtaAddress_phiBendLocal->Fill(address & 0x3f);
                  mismatchGlobalEtaAddress_phiLocal->Fill((address >> 6) & 0x3);
                  mismatchGlobalEtaAddress_wireGroup->Fill((address >> 8) & 0x7f);
                  mismatchGlobalEtaAddress_cscId->Fill((address >> 15) & 0xf);
                }
              
              compareGlobalEtaOffDiagonal->Fill(LUT_1,LUT_2);
            }
          
          differenceGlobalEta->Fill(LUT_1 - LUT_2);
          differenceGlobalEta_etaGlobal->Fill((LUT_1 & 0x7f) - (LUT_2 & 0x7f));
          differenceGlobalEta_phiBend->Fill(((LUT_1 >> 7) & 0x1f) - ((LUT_2 >> 7) & 0x1f));
          
          if((LUT_1 & 0x7f) != (LUT_2 & 0x7f)) // if eta global word does not match 
            {
              compareGlobalEtaOffDiagonal_etaGlobal->Fill((LUT_1 & 0x7f), (LUT_2 & 0x7f));
            }
          if((LUT_1 & 0xf80) != (LUT_2 & 0xf80)) // if phi bend word does not match
            {
              compareGlobalEtaOffDiagonal_phiBend->Fill(((LUT_1 >> 7) & 0x1f), ((LUT_2 >> 7) & 0x1f));
            }
          
          InputVsOutputGlobalEta_1->Fill(address,LUT_1);
          InputVsOutputGlobalEta_2->Fill(address,LUT_2);
        }
    }
  
  // test Pt
  if(doPt)
    {
      for(unsigned int address = 0; address < 1<<CSCBitWidths::kPtAddressWidth; ++address)
        {
          unsigned short LUT_1, LUT_2;
          ptdat out_1, out_2;
          bool legalAddress = true;
          
          try
          {
            out_1 = PtLUT_1->Pt(address);
            LUT_1 = out_1.toint();
          }
          catch(...)
          {
            LUT_1 = 0;
          }
          try
          {
            out_2 = PtLUT_2->Pt(address);
            LUT_2 = out_2.toint();
          }
          catch(...)
          {
            LUT_2 = 0;
          }
          
          if(false) // no cut on pt  address yet
            legalAddress=false;
          
          comparePt->Fill(LUT_1, LUT_2);
          comparePt_front->Fill( (LUT_1 & 0xff), (LUT_2 & 0xff));
          comparePt_rear->Fill( ((LUT_1 & 0xff00)>>8), ((LUT_2 & 0xff00)>>8));
          
          if(LUT_1 != LUT_2)
            {
              mismatchPtAddress->Fill(address);
              if(legalAddress)
                {
                  if(((address & 0xf0000) >> 16) == 0  ||
                     ((address & 0xf0000) >> 16) == 1  ||
                     ((address & 0xf0000) >> 16) == 6  ||
                     ((address & 0xf0000) >> 16) == 7  ||
                     ((address & 0xf0000) >> 16) == 8  ||
                     ((address & 0xf0000) >> 16) == 9  ||
                     ((address & 0xf0000) >> 16) == 10 ||
                     ((address & 0xf0000) >> 16) == 14 ||
                     ((address & 0xf0000) >> 16) == 15 ) // if only 2 stations, use delta12phi and delta23phi separately
                    {
                      mismatchPtAddress_delta12phi->Fill(address & 0xff);
                      mismatchPtAddress_delta23phi->Fill((address & 0xf00) >> 8);
                    }
                  else if(((address & 0xf0000) >> 16) == 2  ||
                          ((address & 0xf0000) >> 16) == 3  ||
                          ((address & 0xf0000) >> 16) == 4  ||
                          ((address & 0xf0000) >> 16) == 5  ||
                          ((address & 0xf0000) >> 16) == 11 ||
                          ((address & 0xf0000) >> 16) == 12 ||
                          ((address & 0xf0000) >> 16) == 13 ) // if 3 stations, use whole deltaPhi word
                    {
                      mismatchPtAddress_deltaPhi->Fill(address & 0xfff);
                    }
                  mismatchPtAddress_eta->Fill((address & 0xf000) >> 12);
                  mismatchPtAddress_mode->Fill((address & 0xf0000) >> 16);
                  mismatchPtAddress_sign->Fill((address & 0x100000) >> 20);
                }
              
              comparePtOffDiagonal->Fill(LUT_1,LUT_2);
            }
          
          differencePt->Fill(LUT_1 - LUT_2);
          differencePt_front->Fill( (LUT_1 & 0xff) - (LUT_2 & 0xff));
          differencePt_rear->Fill( ((LUT_1 & 0xff00)>>8) - ((LUT_2 & 0xff00)>>8));
          
          if((LUT_1 & 0xff) != (LUT_2 & 0xff)) // if Pt front word does not match
            {
              comparePtOffDiagonal_front->Fill((LUT_1 & 0xff), (LUT_2 & 0xff));
            }
          if(LUT_1 & 0xff00 != LUT_2 & 0xff00) // if Pt rear word does not match
            {
              comparePtOffDiagonal_rear->Fill(((LUT_1 & 0xff00)>>8), ((LUT_2 & 0xff00)>>8));
            }
          
          InputVsOutputPt_1->Fill(address, LUT_1);
          InputVsOutputPt_2->Fill(address, LUT_2);
        }
    }
}

double CSCCompareLUTs::getGlobalEtaValue(const int _endcap, const int _sector, const int _subsector, const int _station,
					 const unsigned& thecscid, const unsigned& thewire_group, const unsigned& thephi_local) const
{
  double result = 0.0;
  unsigned wire_group = thewire_group;
  int cscid = thecscid;
  unsigned phi_local = thephi_local;

  // Flag to be set if one wants to apply phi corrections ONLY in ME1/1.
  // Turn it into a parameter?
  bool me1ir_only = false;

  if(cscid < CSCTriggerNumbering::minTriggerCscId() ||
     cscid > CSCTriggerNumbering::maxTriggerCscId()) {
    //edm::LogWarning("CSCSectorReceiverLUT|getEtaValue")
    // << " warning: cscId " << cscid
    // << " is out of bounds [1-" << CSCTriggerNumbering::maxTriggerCscId()
    // << "]\n";
      cscid = CSCTriggerNumbering::maxTriggerCscId();
    }

  CSCTriggerGeomManager* thegeom = CSCTriggerGeometry::get();
  CSCLayerGeometry* layerGeom = NULL;
  const unsigned numBins = 1 << 2; // 4 local phi bins

  if(phi_local > numBins - 1) {
    //edm::LogWarning("CSCSectorReceiverLUT|getEtaValue")
    //<< "warning: phiL " << phi_local
    //<< " is out of bounds [0-" << numBins - 1 << "]\n";
      phi_local = numBins - 1;
  }
  try
    {
      const CSCChamber* thechamber = thegeom->chamber(_endcap,_station,_sector,_subsector,cscid);
      if(thechamber) {
	layerGeom = const_cast<CSCLayerGeometry*>(thechamber->layer(CSCConstants::KEY_ALCT_LAYER)->geometry());
	const unsigned nWireGroups = layerGeom->numberOfWireGroups();

	// Check wire group numbers; expect them to be counted from 0, as in
	// CorrelatedLCTDigi class.
	if (wire_group >= nWireGroups) {
	  //edm::LogWarning("CSCSectorReceiverLUT|getEtaValue")
	  // << "warning: wireGroup " << wire_group
	  // << " is out of bounds [0-" << nWireGroups << ")\n";
	  wire_group = nWireGroups - 1;
	}
	// Convert to [1; nWireGroups] range used in geometry methods.
	wire_group += 1;

	// If me1ir_only is set, apply phi corrections only in ME1/1.
	if (me1ir_only &&
	    (_station != 1 ||
	     CSCTriggerNumbering::ringFromTriggerLabels(_station, cscid) != 1))
	  {
	    result = thechamber->layer(CSCConstants::KEY_ALCT_LAYER)->centerOfWireGroup(wire_group).eta();
	  }
	else {
	  const unsigned nStrips = layerGeom->numberOfStrips();
	  const unsigned nStripsPerBin = CSCConstants::MAX_NUM_STRIPS/numBins;
	  /**
	   * Calculate Eta correction
	   */

	  // Check that no strips will be left out.
	  //if (nStrips%numBins != 0 || CSCConstants::MAX_NUM_STRIPS%numBins != 0)
	  //edm::LogWarning("CSCSectorReceiverLUT")
	  //  << "getGlobalEtaValue warning: number of strips "
	  //  << nStrips << " (" << CSCConstants::MAX_NUM_STRIPS
	  //  << ") is not divisible by numBins " << numBins
	  //  << " Station " << _station << " sector " << _sector
	  //  << " subsector " << _subsector << " cscid " << cscid << "\n";

	  unsigned    maxStripPrevBin = 0, maxStripThisBin = 0;
	  unsigned    correctionStrip;
	  LocalPoint  lPoint;
	  GlobalPoint gPoint;
	  // Bins phi_local and find the the middle strip for each bin.
	  maxStripThisBin = nStripsPerBin * (phi_local+1);
	  if (maxStripThisBin <= nStrips) {
	    correctionStrip = nStripsPerBin/2 * (2*phi_local+1);
	  }
	  else {
	    // If the actual number of strips in the chamber is smaller than
	    // the number of strips corresponding to the right edge of this phi
	    // local bin, we take the middle strip between number of strips
	    // at the left edge of the bin and the actual number of strips.
	    maxStripPrevBin = nStripsPerBin * phi_local;
	    correctionStrip = (nStrips+maxStripPrevBin)/2;
	  }

	  lPoint = layerGeom->stripWireGroupIntersection(correctionStrip, wire_group);
	  gPoint = thechamber->layer(CSCConstants::KEY_ALCT_LAYER)->surface().toGlobal(lPoint);

	  // end calc of eta correction.
	  result = gPoint.eta();
	}
      }
    }
  catch (cms::Exception &e)
    {
      LogDebug("CSCSectorReceiver|OutofBoundInput") << e.what();
    }

  return std::fabs(result);
}


std::string CSCCompareLUTs::encodeFileIndex(int _endcap, int _sector, int _station, int _subsector) const 
{
  std::string fileName = "";
  if (_subsector == 1) fileName += "1a";
  if (_subsector == 2) fileName += "1b";
  fileName += "End";
  if (_endcap == 1) fileName += "1";
  else              fileName += "2";
  fileName += "Sec";
  if      (_sector == 1) fileName += "1";
  else if (_sector == 2) fileName += "2";
  else if (_sector == 3) fileName += "3";
  else if (_sector == 4) fileName += "4";
  else if (_sector == 5) fileName += "5";
  else if (_sector == 6) fileName += "6";
  fileName += "LUT";
  return fileName;
}

std::string CSCCompareLUTs::encodeFileIndex(int _endcap, int _sector, int _station) const 
{
  std::string fileName = "";
  if (_station == 2) fileName += "2";
  else if (_station == 3) fileName += "3";
  else if (_station == 4) fileName += "4";
  fileName += "End";
  if (_endcap == 1) fileName += "1";
  else              fileName += "2";
  fileName += "Sec";
  if      (_sector == 1) fileName += "1";
  else if (_sector == 2) fileName += "2";
  else if (_sector == 3) fileName += "3";
  else if (_sector == 4) fileName += "4";
  else if (_sector == 5) fileName += "5";
  else if (_sector == 6) fileName += "6";
  fileName += "LUT";
  return fileName;
}
