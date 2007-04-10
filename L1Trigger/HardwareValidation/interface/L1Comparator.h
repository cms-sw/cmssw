#ifndef L1COMPARATOR_H
#define L1COMPARATOR_H

/*\class L1Comparator
 *\description L1 trigger data|emulation comparison and validation
 *\author Nuno Leonardo (CERN)
 *\date 07.02
 */

// system include files
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//rct
#include "DataFormats/L1CaloTrigger/interface/L1CaloCollections.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegionDetId.h"

//gct
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctCollections.h"

//gt
#include "DataFormats/L1GlobalTrigger/interface/L1GlobalTriggerReadoutRecord.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTriggerSetup.h"
#include "L1Trigger/GlobalTrigger/interface/L1GlobalTrigger.h"

#include "L1Trigger/HardwareValidation/interface/DEcompare.h"

enum compareMode {RCT=0, GCT, GT};

template <class T> class DEcompare;
 

class L1Comparator : public edm::EDAnalyzer {
public:
  explicit L1Comparator(const edm::ParameterSet&);
  ~L1Comparator();
  
private:
  virtual void beginJob(const edm::EventSetup&);
  virtual void analyze (const edm::Event&, const edm::EventSetup&);
  virtual void endJob();
  
  /// member data
  string RCT_data_Label_;
  string RCT_emul_Label_;
  string GCT_data_Label_;
  string GCT_emul_Label_;
  string GT_data_Label_;
  string GT_emul_Label_;

  string dumpFileName;
  ofstream dumpFile;

  bool doRct_;
  bool doGct_;
  bool doGt_;

  bool rct_match;
  bool gct_match;
  bool gt_match;
  bool evt_match;
  bool all_match;

  bool ReadCollections();

  ///DEBUG&alternatives
  bool dumpCandidate (L1CaloEmCand&, L1CaloEmCand&, ostream& s=std::cout);
  bool dumpCandidate (const L1GctEmCand&, const L1GctEmCand&, ostream& s=std::cout);
  bool compareCollections(edm::Handle<L1CaloEmCollection>    data, edm::Handle<L1CaloEmCollection>    emul);
  bool compareCollections(edm::Handle<L1GctEmCandCollection> data, edm::Handle<L1GctEmCandCollection> emul);
  bool compareCollections(edm::Handle<L1GlobalTriggerReadoutRecord> data, edm::Handle<L1GlobalTriggerReadoutRecord> emul);
  template <class T> bool CompareCollections( edm::Handle<T> data, edm::Handle<T> emul);
};

#endif
