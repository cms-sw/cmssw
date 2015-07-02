#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"


class PhoMVACut : public CutApplicatorWithEventContentBase {
public:
  PhoMVACut(const edm::ParameterSet& c);
  
  result_type operator()(const reco::PhotonPtr&) const override final;

  void setConsumes(edm::ConsumesCollector&) override final;
  void getEventContent(const edm::EventBase&) override final;

  double value(const reco::CandidatePtr& cand) const override final;

  CandidateType candidateType() const override final { 
    return PHOTON; 
  }

private:

  // Cut values
  const std::vector<double> _mvaCutValues;

  // Pre-computed MVA value map
  edm::Handle<edm::ValueMap<float> > _mvaValueMap;
  edm::Handle<edm::ValueMap<int> > _mvaCategoriesMap;

};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
		  PhoMVACut,
		  "PhoMVACut");

PhoMVACut::PhoMVACut(const edm::ParameterSet& c) :
  CutApplicatorWithEventContentBase(c),
  _mvaCutValues(c.getParameter<std::vector<double> >("mvaCuts"))
{
  edm::InputTag mvaValTag = c.getParameter<edm::InputTag>("mvaValueMapName");
  contentTags_.emplace("mvaVal",mvaValTag);
  
  edm::InputTag mvaCatTag = c.getParameter<edm::InputTag>("mvaCategoriesMapName");
  contentTags_.emplace("mvaCat",mvaCatTag);
  
}

void PhoMVACut::setConsumes(edm::ConsumesCollector& cc) {

  auto mvaVal = 
    cc.consumes<edm::ValueMap<float> >(contentTags_["mvaVal"]);
  contentTokens_.emplace("mvaVal",mvaVal);

  auto mvaCat = 
    cc.consumes<edm::ValueMap<int> >(contentTags_["mvaCat"]);
  contentTokens_.emplace("mvaCat",mvaCat);
}

void PhoMVACut::getEventContent(const edm::EventBase& ev) {  

  ev.getByLabel(contentTags_["mvaVal"],_mvaValueMap);
  ev.getByLabel(contentTags_["mvaCat"],_mvaCategoriesMap);
}

CutApplicatorBase::result_type 
PhoMVACut::
operator()(const reco::PhotonPtr& cand) const{  

  // in case we are by-value
  const std::string& inst_name = contentTags_.find("mvaVal")->second.instance();
  edm::Ptr<pat::Photon> pat(cand);

  // Find the cut value
  const int iCategory = _mvaCategoriesMap.isValid() ? (*_mvaCategoriesMap)[cand] : pat->userInt( inst_name + std::string("Category") );
  if( iCategory >= (int)(_mvaCutValues.size()) )
    throw cms::Exception(" Error in MVA categories: ")
      << " found a particle with a category larger than max configured " << std::endl;
  const float cutValue = _mvaCutValues[iCategory];

  // Look up the MVA value for this particle
  const float mvaValue = _mvaValueMap.isValid() ? (*_mvaValueMap)[cand] : pat->userFloat( inst_name + std::string("Value") );

  // Apply the cut and return the result
  return mvaValue > cutValue;
}

double PhoMVACut::value(const reco::CandidatePtr& cand) const {
  
  // in case we are by-value
  const std::string& inst_name =contentTags_.find("mvaVal")->second.instance();
  edm::Ptr<pat::Photon> pat(cand);

  const float mvaValue = _mvaValueMap.isValid() ? (*_mvaValueMap)[cand] : pat->userFloat( inst_name + std::string("Value") );
  return mvaValue;
}
