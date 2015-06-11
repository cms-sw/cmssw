#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"


class GsfEleMVACut : public CutApplicatorWithEventContentBase {
public:
  GsfEleMVACut(const edm::ParameterSet& c);
  
  result_type operator()(const reco::GsfElectronPtr&) const override final;

  void setConsumes(edm::ConsumesCollector&) override final;
  void getEventContent(const edm::EventBase&) override final;

  CandidateType candidateType() const override final { 
    return ELECTRON; 
  }

private:

  // Cut values
  const std::vector<double> _mvaCutValues;

  // Pre-computed MVA value map
  edm::Handle<edm::ValueMap<float> > _mvaValueMap;
  edm::Handle<edm::ValueMap<int> > _mvaCategoriesMap;

};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  GsfEleMVACut,
		  "GsfEleMVACut");

GsfEleMVACut::GsfEleMVACut(const edm::ParameterSet& c) :
  CutApplicatorWithEventContentBase(c),
  _mvaCutValues(c.getParameter<std::vector<double> >("mvaCuts"))
{
  edm::InputTag mvaValTag = c.getParameter<edm::InputTag>("mvaValueMapName");
  contentTags_.emplace("mvaVal",mvaValTag);
  
  edm::InputTag mvaCatTag = c.getParameter<edm::InputTag>("mvaCategoriesMapName");
  contentTags_.emplace("mvaCat",mvaCatTag);
  
}

void GsfEleMVACut::setConsumes(edm::ConsumesCollector& cc) {

  auto mvaVal = 
    cc.consumes<edm::ValueMap<float> >(contentTags_["mvaVal"]);
  contentTokens_.emplace("mvaVal",mvaVal);

  auto mvaCat = 
    cc.consumes<edm::ValueMap<int> >(contentTags_["mvaCat"]);
  contentTokens_.emplace("mvaCat",mvaCat);
}

void GsfEleMVACut::getEventContent(const edm::EventBase& ev) {  

  ev.getByLabel(contentTags_["mvaVal"],_mvaValueMap);
  ev.getByLabel(contentTags_["mvaCat"],_mvaCategoriesMap);
}

CutApplicatorBase::result_type 
GsfEleMVACut::
operator()(const reco::GsfElectronPtr& cand) const{  

  // Find the cut value
  const int iCategory = (*_mvaCategoriesMap)[cand];
  if( iCategory >= (int)(_mvaCutValues.size()) )
    throw cms::Exception(" Error in MVA categories: ")
      << " found a particle with a category larger than max configured " << std::endl;
  const float cutValue = _mvaCutValues[iCategory];

  // Look up the MVA value for this particle
  const float mvaValue = (*_mvaValueMap)[cand];

  // Apply the cut and return the result
  return mvaValue > cutValue;
}
