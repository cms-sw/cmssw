#ifndef DQM_L1TMonitor_L1GtHwValidation_h
#define DQM_L1TMonitor_L1GtHwValidation_h

/**
 * \class L1GtHwValidation
 * 
 * 
 * Description: compare hardware records with emulator records for L1 GT records.
 *
 * Implementation:
 *    Get the L1 GT records from data and from emulator.   
 *    Compare every board between data and emulator.
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date$
 *
 */

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GtObject.h"
#include "CondFormats/L1TObjects/interface/L1GtDefinitions.h"

#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerEvmReadoutRecord.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

// forward declarations
class L1GtfeWord;
class L1GtFdlWord;
class L1GtPsbWord;
class L1TcsWord;
class L1GtTriggerMenu;
class L1GtPrescaleFactors;
class L1GtTriggerMask;

// class declaration

class L1GtHwValidation : public DQMEDAnalyzer {
public:
  explicit L1GtHwValidation(const edm::ParameterSet&);
  ~L1GtHwValidation() override;

private:
  /// compare the GTFE board
  virtual void compareGTFE(const edm::Event&, const edm::EventSetup&, const L1GtfeWord&, const L1GtfeWord&, const int);

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

  /// book all histograms for the module
  //void bookhistograms(DQMStore::IBooker &ibooker);

  /// return true if an algorithm has a condition of that category
  /// for CondNull, it returns always true
  bool matchCondCategory(const L1GtConditionCategory&, const L1GtConditionCategory&);

  /// return true if an algorithm has a condition of that type
  /// for TypeNull, it returns always true
  bool matchCondType(const L1GtConditionType&, const L1GtConditionType&);

  /// return true if an algorithm has a condition containing that object
  /// for ObjNull, it returns always true
  bool matchCondL1GtObject(const std::vector<L1GtObject>&, const L1GtObject&);

  /// exclude from comparison some bits with known disagreement - bit list
  void excludedAlgoList();

  /// exclusion status for algorithm with bit i
  bool excludedAlgo(const int&) const;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

protected:
  void bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) override;
  //virtual void analyze(DQMStore::IBooker &ibooker, const edm::Event&, const edm::EventSetup&);

private:
  /// input tag for the L1 GT hardware DAQ record
  edm::InputTag m_l1GtDataDaqInputTag;

  /// input tag for the L1 GT hardware EVM record
  edm::InputTag m_l1GtDataEvmInputTag;

  /// input tag for the L1 GT emulator DAQ record
  edm::InputTag m_l1GtEmulDaqInputTag;

  /// input tag for the L1 GT emulator EVM record
  edm::InputTag m_l1GtEmulEvmInputTag;

  /// input tag for the L1 GCT hardware record
  edm::InputTag m_l1GctDataInputTag;

  /// directory name for L1Extra plots
  std::string m_dirName;

  /// exclude algorithm triggers from comparison data - emulator by
  /// condition category and / or type
  std::vector<edm::ParameterSet> m_excludeCondCategTypeObject;

  /// exclude algorithm triggers from comparison data - emulator by algorithm name
  std::vector<std::string> m_excludeAlgoTrigByName;

  /// exclude algorithm triggers from comparison data - emulator by algorithm bit number
  std::vector<int> m_excludeAlgoTrigByBit;

private:
  /// excluded condition categories
  std::vector<L1GtConditionCategory> m_excludedCondCategory;

  /// excluded condition types
  std::vector<L1GtConditionType> m_excludedCondType;

  /// excluded L1 GT objects
  std::vector<L1GtObject> m_excludedL1GtObject;

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

  /// prescale factors
  const L1GtPrescaleFactors* m_l1GtPfAlgo;
  unsigned long long m_l1GtPfAlgoCacheID;

  const L1GtPrescaleFactors* m_l1GtPfTech;
  unsigned long long m_l1GtPfTechCacheID;

  const std::vector<std::vector<int> >* m_prescaleFactorsAlgoTrig;
  const std::vector<std::vector<int> >* m_prescaleFactorsTechTrig;

  /// trigger masks
  const L1GtTriggerMask* m_l1GtTmAlgo;
  unsigned long long m_l1GtTmAlgoCacheID;

  const L1GtTriggerMask* m_l1GtTmTech;
  unsigned long long m_l1GtTmTechCacheID;

  std::vector<unsigned int> m_triggerMaskAlgoTrig;
  std::vector<unsigned int> m_triggerMaskTechTrig;

private:
  /// internal members

  bool m_agree;
  bool m_dataOnly;
  bool m_emulOnly;
  bool m_dataOnlyMask;
  bool m_emulOnlyMask;

private:
  static const int TotalBxInEvent = 5;
  static const int NumberOfGtRecords = 2;  // DAQ and EVM

  /// histograms

  /// GTFE
  MonitorElement* m_gtfeDataEmul[NumberOfGtRecords];

  /// FDL (0 for DAQ, 1 for EVM record)
  MonitorElement* m_fdlDataEmul[TotalBxInEvent][NumberOfGtRecords];
  //
  MonitorElement* m_fdlDataAlgoDecision[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlDataAlgoDecisionPrescaled[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlDataAlgoDecisionUnprescaled[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlDataAlgoDecisionMask[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlDataAlgoDecision_NoMatch[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlDataAlgoDecisionPrescaled_NoMatch[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlDataAlgoDecisionUnprescaled_NoMatch[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlDataAlgoDecisionMask_NoMatch[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlDataAlgoDecisionPrescaledMask_NoMatch[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlDataAlgoDecisionUnprescaledMask_NoMatch[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlDataAlgoDecision_Err[NumberOfGtRecords];

  MonitorElement* m_fdlEmulAlgoDecision[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlEmulAlgoDecisionPrescaled[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlEmulAlgoDecisionUnprescaled[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlEmulAlgoDecisionMask[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlEmulAlgoDecision_NoMatch[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlEmulAlgoDecisionPrescaled_NoMatch[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlEmulAlgoDecisionUnprescaled_NoMatch[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlEmulAlgoDecisionMask_NoMatch[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlEmulAlgoDecisionPrescaledMask_NoMatch[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlEmulAlgoDecisionUnprescaledMask_NoMatch[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlEmulAlgoDecision_Err[NumberOfGtRecords];

  //
  MonitorElement* m_fdlDataEmulAlgoDecision[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlDataEmulAlgoDecisionPrescaled[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlDataEmulAlgoDecisionUnprescaled[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlDataEmulAlgoDecisionUnprescaledAllowed[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlDataEmulAlgoDecisionMask[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlDataEmulAlgoDecision_Err[NumberOfGtRecords];
  MonitorElement* m_fdlDataEmul_Err[NumberOfGtRecords];

  //
  MonitorElement* m_fdlDataTechDecision[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlDataTechDecisionMask[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlDataTechDecision_Err[NumberOfGtRecords];

  MonitorElement* m_fdlEmulTechDecision[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlEmulTechDecisionMask[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlEmulTechDecision_Err[NumberOfGtRecords];

  MonitorElement* m_fdlDataEmulTechDecision[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlDataEmulTechDecisionMask[TotalBxInEvent][NumberOfGtRecords];
  MonitorElement* m_fdlDataEmulTechDecision_Err[NumberOfGtRecords];

  MonitorElement* m_excludedAlgorithmsAgreement;

  /// PSB

  // FIXME add PSB comparison

  /// ErrorFlag a la HardwareValidation
  MonitorElement* m_gtErrorFlag;

  ///
  int m_nrEvJob;
  int m_nrEvRun;

  std::vector<int> m_excludedAlgoList;

  //define Token(-s)
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> m_l1GtDataDaqInputToken_;
  edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> m_l1GtEmulDaqInputToken_;
  edm::EDGetTokenT<L1GlobalTriggerEvmReadoutRecord> m_l1GtDataEvmInputToken_;
  edm::EDGetTokenT<L1GlobalTriggerEvmReadoutRecord> m_l1GtEmulEvmInputToken_;
};

#endif /*DQM_L1TMonitor_L1GtHwValidation_h*/
