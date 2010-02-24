#ifndef HLTcore_HLTPrescaleRecorder_h
#define HLTcore_HLTPrescaleRecorder_h

/** \class HLTPrescaleRecorder
 *
 *  
 *  This class is an EDProducer making the HLTPrescaleTable object
 *
 *  $Date: 2010/02/16 10:24:52 $
 *  $Revision: 1.12 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/PrescaleService/interface/PrescaleService.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/HLTReco/interface/HLTPrescaleTable.h"

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

  /// (Single) source 0:PrescaleService, 1:Run, 2:Lumi, 3:Event
  int src_;

  /// (Multiple) Destinations
  bool run_;
  bool lumi_;
  bool event_;

  /// prescale service
  edm::service::PrescaleService* ps_;

  /// InputTag and Handle for existing HLT object
  edm::InputTag hltInputTag_;
  edm::Handle<trigger::HLTPrescaleTable> hltHandle_;

  /// payload HLT object
  trigger::HLTPrescaleTable hlt_;

};
#endif
