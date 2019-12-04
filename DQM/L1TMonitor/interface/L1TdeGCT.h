#ifndef L1TdeGCT_H
#define L1TdeGCT_H

/*\class L1TdeGCT
 *\description GCT data|emulation comparison DQM interface 
               produces expert level DQM monitorable elements
 *\author N.Leonardo
 *\date 08.09
 */

// system, common includes
#include <memory>
#include <string>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
// dqm includes
#include "DQMServices/Core/interface/DQMStore.h"
// l1 dataformats, d|e record includes
#include "L1Trigger/HardwareValidation/interface/DEtrait.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

class L1TdeGCT : public DQMEDAnalyzer {
public:
  explicit L1TdeGCT(const edm::ParameterSet&);
  ~L1TdeGCT() override;

protected:
  void bookHistograms(DQMStore::IBooker& ibooker, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  // input d|e record
  edm::EDGetTokenT<L1DataEmulRecord> DEsource_;
  bool hasRecord_;

  // debug verbose level
  int verbose_;
  int verbose() { return verbose_; }

  // root output file name
  std::string histFile_;

  // dqm histogram folder
  std::string histFolder_;

  bool m_stage1_layer2_;

  // dqm common
  bool monitorDaemon_;

  // (em) iso, no-iso, (jets) cen, for, tau & energy sums.
  static const int nGctColl_ = dedefs::GCThfbit - dedefs::GCTisolaem + 1;
  static const int nStage1Layer2Coll_ = dedefs::GCTisotaujets - dedefs::GCTisolaem + 1;

  // counters
  int colCount[nGctColl_];
  int nWithCol[nGctColl_];

  int colCount_stage1Layer2[nStage1Layer2Coll_];
  int nWithCol_stage1Layer2[nStage1Layer2Coll_];

  // Ranges and labels
  const int phiNBins = 18;
  const double phiMinim = -0.5;
  const double phiMaxim = 17.5;
  const int etaNBins = 22;
  const double etaMinim = -0.5;
  const double etaMaxim = 21.5;
  static const int nerr = 5;
  const int nbit = 32;
  std::string cLabel[nGctColl_] = {
      "IsoEM", "NoisoEM", "CenJet", "ForJet", "TauJet", "HT", "MET", "ET", "MHT", "HFSums", "HFCnts"};
  std::string sLabel[nStage1Layer2Coll_] = {
      "IsoEM", "NoisoEM", "CenJet", "ForJet", "TauJet", "HT", "MET", "ET", "MHT", "Stage1HFSums", "HFCnts", "IsoTauJet"};
  std::string errLabel[nerr] = {"Agree", "Loc. Agree", "L.Disagree", "Data only", "Emul only"};

  // MEs
  MonitorElement* sysrates;
  MonitorElement* sysncand[2];

  MonitorElement* errortype[nGctColl_];
  // location
  MonitorElement* etaphi[nGctColl_];
  MonitorElement* eta[nGctColl_];
  MonitorElement* phi[nGctColl_];
  MonitorElement* rnk[nGctColl_];
  MonitorElement* etaData[nGctColl_];
  MonitorElement* phiData[nGctColl_];
  MonitorElement* rnkData[nGctColl_];

  MonitorElement* errortype_stage1layer2[nStage1Layer2Coll_];
  // location
  MonitorElement* etaphi_stage1layer2[nStage1Layer2Coll_];
  MonitorElement* eta_stage1layer2[nStage1Layer2Coll_];
  MonitorElement* phi_stage1layer2[nStage1Layer2Coll_];
  MonitorElement* rnk_stage1layer2[nStage1Layer2Coll_];
  MonitorElement* etaData_stage1layer2[nStage1Layer2Coll_];
  MonitorElement* phiData_stage1layer2[nStage1Layer2Coll_];
  MonitorElement* rnkData_stage1layer2[nStage1Layer2Coll_];

  // trigger data word

  MonitorElement* dword[nGctColl_];
  MonitorElement* eword[nGctColl_];
  MonitorElement* deword[nGctColl_];
  MonitorElement* masked[nGctColl_];
  MonitorElement* dword_stage1layer2[nStage1Layer2Coll_];
  MonitorElement* eword_stage1layer2[nStage1Layer2Coll_];
  MonitorElement* deword_stage1layer2[nStage1Layer2Coll_];
  MonitorElement* masked_stage1layer2[nStage1Layer2Coll_];

public:
};

#endif
