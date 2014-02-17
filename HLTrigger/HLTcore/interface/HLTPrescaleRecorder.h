#ifndef HLTcore_HLTPrescaleRecorder_h
#define HLTcore_HLTPrescaleRecorder_h

/** \class HLTPrescaleRecorder
 *
 *  
 *  This class is an EDProducer making the HLTPrescaleTable object
 *
 *  $Date: 2010/03/08 17:12:09 $
 *  $Revision: 1.4 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/PrescaleService/interface/PrescaleService.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/HLTReco/interface/HLTPrescaleTable.h"
#include "CondFormats/HLTObjects/interface/HLTPrescaleTableCond.h"

#include<map>
#include<string>
#include<vector>

//
// class declaration
//

class HLTPrescaleRecorder : public edm::EDProducer {

 public:
  explicit HLTPrescaleRecorder(const edm::ParameterSet&);
  ~HLTPrescaleRecorder();
  virtual void beginRun(edm::Run& iRun, const edm::EventSetup& iSetup);
  virtual void endRun(edm::Run& iRun, const edm::EventSetup& iSetup);
  virtual void beginLuminosityBlock(edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup);
  virtual void endLuminosityBlock(edm::LuminosityBlock& iLumi, const edm::EventSetup& iSetup);
  virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup);

 private:

  /// (Single) source: -1:PrescaleServicePSet 0:PrescaleService,
  /// 1:Run, 2:Lumi, 3:Event, 4:CondDB
  int src_;

  /// (Multiple) Destinations
  bool run_;
  bool lumi_;
  bool event_;
  bool condDB_;

  /// Source configs
  /// name of PrescaleServicePSet (src=-1)
  std::string psetName_;
  /// InputTag of HLTPrescaleTable product (src=1,2,3)
  edm::InputTag hltInputTag_;
  /// Tag of DB entry (HLT Key Name) (src=4)
  std::string hltDBTag_;

  /// Prescale service
  edm::service::PrescaleService* ps_;
  /// Pool DB service
  cond::service::PoolDBOutputService* db_;

  /// Handle and ESHandle for existing HLT object
  edm::Handle<trigger::HLTPrescaleTable> hltHandle_;
  edm::ESHandle<trigger::HLTPrescaleTableCond> hltESHandle_;

  /// payload HLT object
  trigger::HLTPrescaleTable hlt_;

};
#endif
