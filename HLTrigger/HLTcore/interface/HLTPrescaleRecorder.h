#ifndef HLTcore_HLTPrescaleRecorder_h
#define HLTcore_HLTPrescaleRecorder_h

/** \class HLTPrescaleRecorder
 *
 *  
 *  This class is an EDProducer making the HLTPrescaleTable object
 *
 *  $Date: 2013/05/17 20:33:54 $
 *  $Revision: 1.6 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
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

class HLTPrescaleRecorder : public edm::one::EDProducer<edm::EndRunProducer,
                                                        edm::EndLuminosityBlockProducer,
                                                        edm::one::WatchRuns,
                                                        edm::one::WatchLuminosityBlocks> {

 public:
  explicit HLTPrescaleRecorder(const edm::ParameterSet&);
  virtual ~HLTPrescaleRecorder();
  virtual void beginRun(edm::Run const& iRun, const edm::EventSetup& iSetup)override final;
  virtual void endRun(edm::Run const& iRun, const edm::EventSetup& iSetup)override final;
  virtual void endRunProduce(edm::Run & iRun, const edm::EventSetup& iSetup)override final;
  virtual void beginLuminosityBlock(edm::LuminosityBlock const& iLumi, const edm::EventSetup& iSetup)override final;
  virtual void endLuminosityBlock(edm::LuminosityBlock const& iLumi, const edm::EventSetup& iSetup)override final;
  virtual void endLuminosityBlockProduce(edm::LuminosityBlock & iLumi, const edm::EventSetup& iSetup)override final;
  virtual void produce(edm::Event& iEvent, const edm::EventSetup& iSetup)override final;

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
