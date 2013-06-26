#ifndef L1COMPARATOR_H
#define L1COMPARATOR_H

/*\class L1Comparator
 *\description L1 trigger data|emulation comparison and validation
 *\author Nuno Leonardo (CERN)
 *\date 07.02
 */

// common/system includes
#include <memory>
#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// l1 dataformats, d|e record includes
#include "L1Trigger/HardwareValidation/interface/DEtrait.h"

// comparator template
#include "L1Trigger/HardwareValidation/interface/DEcompare.h"

// db trigger/subsystem key access
#include "CondFormats/L1TObjects/interface/L1TriggerKey.h"	   
//#include "CondFormats/L1TObjects/interface/L1TriggerKeyList.h"   
#include "CondFormats/DataRecord/interface/L1TriggerKeyRcd.h"	   
//#include "CondFormats/DataRecord/interface/L1TriggerKeyListRcd.h"

template <class T> class DEcompare;

class L1Comparator : public edm::EDProducer {

public:

  explicit L1Comparator(const edm::ParameterSet&);
  ~L1Comparator();
  
private:

  virtual void beginJob(void);
  virtual void beginRun(edm::Run const&, const edm::EventSetup&) override final;
   virtual void produce (edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  template <class T> 
    void process( T const*, T const*, const int, const int);
  template <class T> 
    void process(const edm::Handle<T> data, const edm::Handle<T> emul, 
		 const int sys, const int cid) {
    if(data.isValid()&&emul.isValid())
      process(data.product(),emul.product(),sys, cid);
  }

  template <class T> bool CompareCollections(edm::Handle<T> data, edm::Handle<T> emul);
  template <class T> bool dumpCandidate(const T& dt, const T& em, std::ostream& s);

  int verbose() {return verbose_;}

 private:

  int nevt_;
  int evtNum_;
  int runNum_;
  int verbose_;
  bool dumpEvent_;

  edm::InputTag m_DEsource[dedefs::DEnsys][4];
  bool m_doSys[dedefs::DEnsys];
  std::string m_dumpFileName;
  std::ofstream m_dumpFile;
  int m_dumpMode;
  bool m_match;
  bool DEmatchEvt[dedefs::DEnsys]; 
  int DEncand[dedefs::DEnsys][2];
  L1DEDigiCollection m_dedigis;

};

#endif
