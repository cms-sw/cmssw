#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonSelectors.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"


class HLTDQMMuonSelector : public edm::stream::EDProducer<> {
public:
  explicit HLTDQMMuonSelector(const edm::ParameterSet& config);
  void produce(edm::Event&, edm::EventSetup const&)override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
private:

  enum class MuonSelectionType {
    Tight,Medium,Loose,Soft,HighPt,None
  };
  
  static MuonSelectionType convertToEnum(const std::string& val);
  bool passMuonSel(const reco::Muon& muon,const reco::Vertex& vertex)const;
  

  edm::EDGetTokenT<reco::MuonCollection> muonToken_;
  edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
  StringCutObjectSelector<reco::Muon,true> selection_;
  
  MuonSelectionType muonSelType_;
  
  
};


HLTDQMMuonSelector::HLTDQMMuonSelector(const edm::ParameterSet& config):
  muonToken_(consumes<reco::MuonCollection>(config.getParameter<edm::InputTag>("objs"))),
  vtxToken_(consumes<reco::VertexCollection>(config.getParameter<edm::InputTag>("vertices"))),
  selection_(config.getParameter<std::string>("selection")),
  muonSelType_(convertToEnum(config.getParameter<std::string>("muonSelectionType")))
{ 
  produces<edm::ValueMap<bool> >(); 
}

void HLTDQMMuonSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions)
{
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("objs", edm::InputTag("muons"));
  desc.add<edm::InputTag>("vertices", edm::InputTag("offlinePrimaryVertices"));
  desc.add<std::string>("selection","et > 5");
  desc.add<std::string>("muonSelectionType","tight");
  descriptions.add("hltDQMMuonSelector", desc);
}


void HLTDQMMuonSelector::produce(edm::Event& event,const edm::EventSetup& setup)
{
  edm::Handle<reco::MuonCollection> muonHandle;
  event.getByToken(muonToken_,muonHandle);
  
  edm::Handle<reco::VertexCollection> vtxHandle;
  event.getByToken(vtxToken_,vtxHandle);
  
  if(!muonHandle.isValid()) return;
  
  std::vector<bool> selResults;
  for(auto& muon : *muonHandle){
    if(vtxHandle.isValid() && !vtxHandle->empty()){
      selResults.push_back(passMuonSel(muon,vtxHandle->front()) && selection_(muon));
    }else{
      selResults.push_back(false);
    }
  }
  auto valMap = std::make_unique<edm::ValueMap<bool> >();
  edm::ValueMap<bool>::Filler filler(*valMap);
  filler.insert(muonHandle, selResults.begin(), selResults.end());
  filler.fill();
  event.put(std::move(valMap));
}
	       
HLTDQMMuonSelector::MuonSelectionType HLTDQMMuonSelector::convertToEnum(const std::string& val)
{
  const std::vector<std::pair<std::string,MuonSelectionType> > strsToEnums = {
    {"tight",MuonSelectionType::Tight},
    {"medium",MuonSelectionType::Medium},
    {"loose",MuonSelectionType::Loose},
    {"soft",MuonSelectionType::Soft},
    {"highpt",MuonSelectionType::HighPt},
    {"none",MuonSelectionType::None}
  };
  for(const auto& strEnumPair : strsToEnums){
    if(val==strEnumPair.first) return strEnumPair.second;
  }
  std::ostringstream validEnums;
  for(const auto& strEnumPair : strsToEnums) validEnums <<strEnumPair.first<<" ";
  throw cms::Exception("InvalidConfig") << "invalid muonSelectionType "<<val<<", allowed values are "<<validEnums.str();
}

bool HLTDQMMuonSelector::passMuonSel(const reco::Muon& muon,const reco::Vertex& vertex)const
{
  switch(muonSelType_){
  case MuonSelectionType::Tight:
    return muon::isTightMuon(muon,vertex);
  case MuonSelectionType::Medium:
    return muon::isMediumMuon(muon);
  case MuonSelectionType::Loose:
    return muon::isLooseMuon(muon);
  case MuonSelectionType::Soft:
    return muon::isSoftMuon(muon,vertex);
  case MuonSelectionType::HighPt:
    return muon::isHighPtMuon(muon,vertex);
  case MuonSelectionType::None:
    return true;
  default:
    edm::LogError("HLTDQMMuonSelector")<<" inconsistent code, an option has been added to MuonSelectionType without updating HLTDQMMuonSelector::passMuonSel";
    return false;
  }

}
  
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTDQMMuonSelector);
