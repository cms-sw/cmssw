/** \class HLTPrescaler
 *
 *  
 *  See header file for documentation.
 *
 *  $Date: 2007/04/18 07:20:56 $
 *  $Revision: 1.1 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/HLTPrescaler.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HLTPrescaler::HLTPrescaler(edm::ParameterSet const& ps) :
  b_(ps.getParameter<bool>("makeFilterObject")),
  n_(ps.getParameter<unsigned int>("prescaleFactor")),
  o_(ps.getParameter<unsigned int>("eventOffset")),
  count_(0), 
  ps_(0)
{
  if (b_) produces<reco::HLTFilterObjectBase>();
  if (n_==0) n_=1; // accept all!
  count_ = o_;     // event offset

  // get prescale service
  try {
    if(edm::Service<edm::service::PrescaleService>().isAvailable()) {
      ps_ = edm::Service<edm::service::PrescaleService>().operator->();
    } else {
      LogDebug("HLTPrescaler ") << "non available service edm::service::PrescaleService\n";
    }
  }
  catch(...) {
    LogDebug("HLTPrescaler ") << "exception getting service edm::service::PrescaleService\n";
  }

  if (ps_==0) {
    LogDebug("HLTPrescaler ") << "prescale service pointer == 0 - using config default\n";
  } else {
    LogDebug("HLTPrescaler ") << "prescale service pointer != 0 - using service\n";
  }

}
    
HLTPrescaler::~HLTPrescaler()
{
}

bool HLTPrescaler::filter(edm::Event & e, const edm::EventSetup & es)
{
  using namespace std;
  using namespace edm;
  using namespace reco;

  // get prescaler from service 
  if (ps_) {
    int newPrescale(ps_->getPrescale(0,*pathName()));
    if (newPrescale < 0 ) {
      LogDebug("HLTPrescaler ") << "edm::service::PrescaleService for path " << *pathName() << " not found\n";
    } else {
      if (newPrescale>0) n_ = newPrescale;
    }
  }

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
