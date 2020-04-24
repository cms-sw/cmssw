#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Common/interface/ValueMap.h"

template<typename ObjType,typename ObjCollType>
class HLTDQMObjSelector : public edm::stream::EDProducer<> {
public:
  explicit HLTDQMObjSelector(const edm::ParameterSet& config);
  void produce(edm::Event&, edm::EventSetup const&)override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
private:
  edm::EDGetTokenT<ObjCollType> token_;
  StringCutObjectSelector<ObjType,true> selection_;

};

template<typename ObjType,typename ObjCollType>
HLTDQMObjSelector<ObjType,ObjCollType>::HLTDQMObjSelector(const edm::ParameterSet& config):
  token_(consumes<ObjCollType>(config.getParameter<edm::InputTag>("objs"))),
  selection_(config.getParameter<std::string>("selection"))
{ 
  produces<edm::ValueMap<bool> >(); 
}

template<typename ObjType,typename ObjCollType>
void HLTDQMObjSelector<ObjType,ObjCollType>::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("objs", edm::InputTag(""));
  desc.add<std::string>("selection","et > 5");
  descriptions.add("hltDQMObjSelector", desc);
}

template<typename ObjType,typename ObjCollType>
void HLTDQMObjSelector<ObjType,ObjCollType>::produce(edm::Event& event,const edm::EventSetup& setup)
{
  edm::Handle<ObjCollType> handle;
  event.getByToken(token_,handle);
  
  if(!handle.isValid()) return;
  
  std::vector<bool> selResults;
  for(auto& obj : *handle){
    selResults.push_back(selection_(obj));
  }
  auto valMap = std::make_unique<edm::ValueMap<bool> >();
  edm::ValueMap<bool>::Filler filler(*valMap);
  filler.insert(handle, selResults.begin(), selResults.end());
  filler.fill();
  event.put(std::move(valMap));
}


#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
using HLTDQMGsfEleSelector = HLTDQMObjSelector<reco::GsfElectron,reco::GsfElectronCollection>;
DEFINE_FWK_MODULE(HLTDQMGsfEleSelector);
