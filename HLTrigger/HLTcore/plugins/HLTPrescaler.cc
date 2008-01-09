/** \class HLTPrescaler
 *
 *  
 *  See header file for documentation.
 *
 *  $Date: 2008/01/09 13:59:06 $
 *  $Revision: 1.5 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTPrescaler.h"
#include "DataFormats/HLTReco/interface/TriggerFilterObjectWithRefs.h"

#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HLTPrescaler::HLTPrescaler(edm::ParameterSet const& ps) :
  b_(ps.getParameter<bool>("makeFilterObject")),
  n_(ps.getParameter<unsigned int>("prescaleFactor")),
  o_(ps.getParameter<unsigned int>("eventOffset")),
  count_(0), 
  ps_(0),
  moduleLabel_(ps.getParameter<std::string>("@module_label"))
{
  if (b_) produces<trigger::TriggerFilterObjectWithRefs>();
  if (n_==0) n_=1; // accept all!
  count_ = o_;     // event offset

  // get prescale service
  if(edm::Service<edm::service::PrescaleService>().isAvailable()) {
    // isAvailable() throws only if the entire service system is not available
    ps_ = edm::Service<edm::service::PrescaleService>().operator->();
  } else {
    LogDebug("HLTPrescaler ") << "non available service edm::service::PrescaleService.";
    ps_ = 0;
  }

  if (ps_==0) {
    LogDebug("HLTPrescaler ") << "prescale service pointer == 0 - using module config default.";
  } else {
    LogDebug("HLTPrescaler ") << "prescale service pointer != 0 - using prescale service.";
  }

}
    
HLTPrescaler::~HLTPrescaler()
{
}

bool HLTPrescaler::beginLuminosityBlock(edm::LuminosityBlock & lb, edm::EventSetup const& es)
{
  using namespace std;
  using namespace edm;
  using namespace reco;

  LogDebug("HLTPrescaler") << "New LumiBlock: " <<lb.id().luminosityBlock();
  if (ps_) {
    // get prescale value from service 
//  int newPrescale(ps_->getPrescale(lb.id().luminosityBlock(),moduleLabel_));
    int newPrescale(ps_->getPrescale(moduleLabel_));
    LogDebug("HLTPrescaler") << "Returned value: " << newPrescale;
    if (newPrescale < 0 ) {
      LogDebug("HLTPrescaler") << "PrescaleService: no info for module - using module value: " << n_ ;
    } else {
      n_=newPrescale;
      if (n_==0) n_=1; // accept all!
      count_ = o_;     // event offset
    }
  }

  return true;
}

bool HLTPrescaler::filter(edm::Event & e, const edm::EventSetup & es)
{
  using namespace std;
  using namespace edm;
  using namespace reco;
  using namespace trigger;

  // prescaler decision
  ++count_;
  const bool accept(count_%n_ == 0);

  // construct and place filter object if requested
  if (b_) {
    auto_ptr<TriggerFilterObjectWithRefs> 
      filterproduct (new TriggerFilterObjectWithRefs(path(),module()));
    e.put(filterproduct);
  }

  return accept;

}
