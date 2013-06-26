#include "RecoMuon/MuonIsolationProducers/plugins/MuIsoDepositCopyProducer.h"

// Framework
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/RecoCandidate/interface/IsoDeposit.h"
#include "DataFormats/RecoCandidate/interface/IsoDepositFwd.h"
#include "DataFormats/Common/interface/ValueMap.h"


#include "FWCore/Utilities/interface/Exception.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <string>

using namespace edm;
using namespace std;
using namespace reco;


//! constructor with config
MuIsoDepositCopyProducer::MuIsoDepositCopyProducer(const ParameterSet& par) :
  theConfig(par),
  theInputTags(par.getParameter<std::vector<edm::InputTag> >("inputTags")),
  theDepositNames(par.getParameter<std::vector<std::string> >("depositNames"))

{
  LogDebug("RecoMuon|MuonIsolation")<<" MuIsoDepositCopyProducer CTOR";
  if (theInputTags.size() != theDepositNames.size()){
    throw  cms::Exception("MuIsoDepositCopyProducer constructor")<<"the sizes of input/output vectors don't match";
  }
  
  for (unsigned int i = 0; i < theDepositNames.size(); ++i){
    std::string alias = theConfig.getParameter<std::string>("@module_label");
    if (theDepositNames[i] != "") alias += "_" + theDepositNames[i];
    produces<reco::IsoDepositMap>(theDepositNames[i]).setBranchAlias(alias);
  }
}

//! destructor
MuIsoDepositCopyProducer::~MuIsoDepositCopyProducer(){
  LogDebug("RecoMuon/MuIsoDepositCopyProducer")<<" MuIsoDepositCopyProducer DTOR";
}

//! build deposits
void MuIsoDepositCopyProducer::produce(Event& event, const EventSetup& eventSetup){
  std::string metname = "RecoMuon|MuonIsolationProducers|MuIsoDepositCopyProducer";

  LogDebug(metname)<<" Muon Deposit producing..."
		   <<" BEGINING OF EVENT " <<"================================";

  LogTrace(metname)<<" Taking the inputs: ";

  for (unsigned int iDep = 0; iDep < theInputTags.size(); ++iDep){
    Handle<reco::IsoDepositMap > inDep;
    event.getByLabel(theInputTags[iDep], inDep);

    std::auto_ptr<reco::IsoDepositMap> outDep(new reco::IsoDepositMap(*inDep));
    event.put(outDep, theDepositNames[iDep]);
  }//! end iDep

  LogTrace(metname) <<" END OF EVENT " <<"================================";
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(MuIsoDepositCopyProducer);
