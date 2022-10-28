#ifndef GlobalTriggerAnalyzer_L1GtDataEmulAnalyzer_h
#define GlobalTriggerAnalyzer_L1GtDataEmulAnalyzer_h

/**
 * \class L1GtDataEmulAnalyzer
 * 
 * 
 * Description: compare hardware records with emulator records for L1 GT record.  
 *
 * Implementation:
 *    Get the L1 GT records from data and from emulator.   
 *    Compare every board between data and emulator.
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 *
 */

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

// forward declarations
class L1GtfeWord;
class L1GtFdlWord;
class L1GtPsbWord;
class L1TcsWord;
class L1GtTriggerMenu;
class L1GtTriggerMask;
class L1GtTriggerMenuRcd;
class L1GtTriggerMaskAlgoTrigRcd;
class L1GtTriggerMaskTechTrigRcd;

class TH1F;
class TH1D;
class TH2D;
class TTree;

// class declaration

class L1GtDataEmulAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit L1GtDataEmulAnalyzer(const edm::ParameterSet&);
  ~L1GtDataEmulAnalyzer() override;

private:
  void beginJob() override;

  /// compare the GTFE board
  virtual void compareGTFE(const edm::Event&, const edm::EventSetup&, const L1GtfeWord&, const L1GtfeWord&);

  /// compare the FDL board
  virtual void compareFDL(const edm::Event&, const edm::EventSetup&, const L1GtFdlWord&, const L1GtFdlWord&, const int);

  /// compare the PSB board
  virtual void comparePSB(const edm::Event&, const edm::EventSetup&, const L1GtPsbWord&, const L1GtPsbWord&);

  /// compare the TCS board
  virtual void compareTCS(const edm::Event&, const edm::EventSetup&, const L1TcsWord&, const L1TcsWord&);

  /// L1 GT DAQ record comparison
  virtual void compareDaqRecord(const edm::Event&, const edm::EventSetup&);

  /// L1 GT EVM record comparison
  virtual void compareEvmRecord(const edm::Event&, const edm::EventSetup&);

  /// compare the GCT collections obtained from L1 GT PSB with the input
  /// GCT collections
  virtual void compareGt_Gct(const edm::Event&, const edm::EventSetup&);

  /// analyze each event
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  /// book all histograms for the module
  void bookHistograms();

  /// end of job
  void endJob() override;

private:
  /// input tag for the L1 GT hardware DAQ/EVM record
  edm::InputTag m_l1GtDataInputTag;

  /// input tag for the L1 GT emulator DAQ/EVM record
  edm::InputTag m_l1GtEmulInputTag;

  /// input tag for the L1 GCT hardware record
  edm::InputTag m_l1GctDataInputTag;

private:
  /// an output stream to print into
  /// it can then be directed to whatever log level is desired
  std::ostringstream m_myCoutStream;

  /// counters
  int m_nrDataEventError;
  int m_nrEmulEventError;

  // cached stuff

  /// trigger menu
  const L1GtTriggerMenu* m_l1GtMenu;
  unsigned long long m_l1GtMenuCacheID;

  /// trigger masks
  const L1GtTriggerMask* m_l1GtTmAlgo;
  unsigned long long m_l1GtTmAlgoCacheID;

  const L1GtTriggerMask* m_l1GtTmTech;
  unsigned long long m_l1GtTmTechCacheID;

  std::vector<unsigned int> m_triggerMaskAlgoTrig;
  std::vector<unsigned int> m_triggerMaskTechTrig;

private:
  /// histograms

  /// GTFE
  TH1F* m_gtfeDataEmul;

  static constexpr int TotalBxInEvent = 5;

  /// FDL (0 for DAQ, 1 for EVM record)
  TH1F* m_fdlDataEmul[TotalBxInEvent][2];

  TH1F* m_fdlDataAlgoDecision[TotalBxInEvent][2];
  TH1F* m_fdlEmulAlgoDecision[TotalBxInEvent][2];

  TH1F* m_fdlDataAlgoDecisionMask[TotalBxInEvent][2];
  TH1F* m_fdlEmulAlgoDecisionMask[TotalBxInEvent][2];

  TH1F* m_fdlDataEmulAlgoDecision[TotalBxInEvent][2];
  TH1F* m_fdlDataEmulAlgoDecisionMask[TotalBxInEvent][2];

  TH1F* m_fdlDataTechDecision[TotalBxInEvent][2];
  TH1F* m_fdlEmulTechDecision[TotalBxInEvent][2];

  TH1F* m_fdlDataTechDecisionMask[TotalBxInEvent][2];
  TH1F* m_fdlEmulTechDecisionMask[TotalBxInEvent][2];

  TH1F* m_fdlDataEmulTechDecision[TotalBxInEvent][2];
  TH1F* m_fdlDataEmulTechDecisionMask[TotalBxInEvent][2];

  TH1F* m_fdlDataEmul_Err[2];

  TH1F* m_fdlDataAlgoDecision_Err[2];
  TH1F* m_fdlEmulAlgoDecision_Err[2];
  TH1F* m_fdlDataEmulAlgoDecision_Err[2];

  TH1F* m_fdlDataTechDecision_Err[2];
  TH1F* m_fdlEmulTechDecision_Err[2];
  TH1F* m_fdlDataEmulTechDecision_Err[2];

  /// PSB

  edm::ESGetToken<L1GtTriggerMenu, L1GtTriggerMenuRcd> m_l1GtMenuToken;
  edm::ESGetToken<L1GtTriggerMask, L1GtTriggerMaskAlgoTrigRcd> m_l1GtTmAlgoToken;
  edm::ESGetToken<L1GtTriggerMask, L1GtTriggerMaskTechTrigRcd> m_l1GtTmTechToken;
};

#endif /*GlobalTriggerAnalyzer_L1GtDataEmulAnalyzer_h*/
