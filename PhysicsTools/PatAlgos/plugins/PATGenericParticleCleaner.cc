//
// $Id: PATGenericParticleCleaner.cc,v 1.3 2008/06/06 14:13:40 gpetrucc Exp $
//

#include "PhysicsTools/PatAlgos/plugins/PATGenericParticleCleaner.h"
#include "DataFormats/PatCandidates/interface/Flags.h"

pat::PATGenericParticleCleaner::PATGenericParticleCleaner(const edm::ParameterSet & iConfig) :
  src_(iConfig.getParameter<edm::InputTag>( "src" )),
  helper_(src_),
  isolator_(iConfig.exists("isolation") ? iConfig.getParameter<edm::ParameterSet>("isolation") : edm::ParameterSet() )
{
  helper_.configure(iConfig);      // learn whether to save good, bad, all, ...
  helper_.registerProducts(*this); // issue the produces<>() commands

  if (iConfig.exists("removeOverlaps")) {
    edm::ParameterSet overlapConf = iConfig.getParameter<edm::ParameterSet>("removeOverlaps");
    overlapHelper_ = pat::helper::OverlapHelper(overlapConf);
  }
}


pat::PATGenericParticleCleaner::~PATGenericParticleCleaner() {
}


void pat::PATGenericParticleCleaner::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {     
  // start a new event
  helper_.newEvent(iEvent);
  if (isolator_.enabled()) isolator_.beginEvent(iEvent);

  for (size_t idx = 0, size = helper_.srcSize(); idx < size; ++idx) {
    // read the source object and clone it
    reco::Candidate * srcGenericParticle = helper_.srcAt(idx).clone();

    // write the object
    size_t selIdx = helper_.addItem(idx, srcGenericParticle);
    
    // test for isolation and set the bit if needed
    if (isolator_.enabled()) {
        uint32_t isolationWord = isolator_.test( helper_.source(), idx );
        helper_.addMark(selIdx, isolationWord);
    }
  }

  if (overlapHelper_.enabled()) {
     typedef pat::helper::OverlapHelper::Result Result;
     std::auto_ptr<Result> result = overlapHelper_.test( iEvent, helper_.selected() );
     for (size_t i = 0, n = helper_.size(); i < n; ++i) {
        helper_.addMark( i, (*result)[i] );
     }
  }

  helper_.done();
  if (isolator_.enabled()) isolator_.endEvent();

}


void pat::PATGenericParticleCleaner::endJob() { 
    edm::LogVerbatim("PATLayer0Summary|PATGenericParticleCleaner") << "PATGenericParticleCleaner end job. \n" <<
            "Input tag was " << src_.encode() <<
            "\nIsolation information:\n" <<
            isolator_.printSummary() <<
            "\nCleaner summary information:\n" <<
            helper_.printSummary();
}

#include "FWCore/Framework/interface/MakerMacros.h"

using namespace pat;
DEFINE_FWK_MODULE(PATGenericParticleCleaner);
