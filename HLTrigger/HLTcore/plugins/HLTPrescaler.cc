/** \class HLTPrescaler
 *
 *  
 *  See header file for documentation.
 *
 *  $Date: 2007/08/02 21:52:06 $
 *  $Revision: 1.2 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTPrescaler.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

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
  if (b_) produces<reco::HLTFilterObjectBase>();
  if (n_==0) n_=1; // accept all!
  count_ = o_;     // event offset

  // get prescale service
  try {
    if(edm::Service<edm::service::PrescaleService>().isAvailable()) {
      ps_ = edm::Service<edm::service::PrescaleService>().operator->();
    } else {
      LogDebug("HLTPrescaler ") << "non available service edm::service::PrescaleService.";
    }
  }
  catch(...) {
    LogDebug("HLTPrescaler ") << "exception getting service edm::service::PrescaleService.";
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

  // prescaler decision
  ++count_;
  const bool accept(count_%n_ == 0);

  // construct and place filter object if requested
  if (b_) {
    auto_ptr<HLTFilterObjectBase> 
      filterproduct (new HLTFilterObjectBase(path(),module()));
    e.put(filterproduct);
  }

  return accept;

}
