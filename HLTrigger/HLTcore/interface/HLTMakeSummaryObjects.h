#ifndef HLTMakeSummaryObjects_h
#define HLTMakeSummaryObjects_h

/** \class HLTMakeSummaryObjects
 *
 *  
 *  This class is an EDProducer making the HLT summary objects (path
 *  objects and global object).
 *
 *  $Date: 2007/06/08 09:58:57 $
 *  $Revision: 1.12 $
 *
 *  \author Martin Grunewald
 *
 */

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/CurrentProcessingContext.h"
#include "FWCore/Framework/interface/Selector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Framework/interface/TriggerNamesService.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

#include "DataFormats/HLTReco/interface/HLTGlobalObject.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"
#include "DataFormats/HLTReco/interface/HLTPathObject.h"
#include<string>
#include<vector>

//
// class declaration
//

class HLTMakeSummaryObjects : public edm::EDProducer {

  public:
    explicit HLTMakeSummaryObjects(const edm::ParameterSet&);
    ~HLTMakeSummaryObjects();
    virtual void produce(edm::Event&, const edm::EventSetup&);

  private:
    /// the pointer to the current TriggerNamesService
    edm::service::TriggerNamesService* tns_;
    /// selector for getMany methods
    edm::ProcessNameSelector selector_;

    /// handles to the various types of filter objects
    std::vector<edm::Handle<reco::HLTFilterObjectBase    > > fob0_;
    std::vector<edm::Handle<reco::HLTFilterObject        > > fob1_;
    std::vector<edm::Handle<reco::HLTFilterObjectWithRefs> > fob2_;
    /// reftobase allowing combined access to these
    std::vector<edm::RefToBase<reco::HLTFilterObjectBase > > fobs_;
    /// pointer to labels of filter modules
    std::vector<const std::string * >                    fobnames_;
    /// vector for path objects to be produced
    std::vector<edm::OrphanHandle<reco::HLTPathObject> >     pobs_;

};

#endif //HLTMakeSummaryObjects_h
