#include "PhysicsTools/SelectorUtils/interface/CutApplicatorWithEventContentBase.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"

#include <cmath>

class GsfEleMVAExpoScalingCut : public CutApplicatorWithEventContentBase {
public:
  GsfEleMVAExpoScalingCut(const edm::ParameterSet& c);

  result_type operator()(const reco::GsfElectronPtr&) const final;

  void setConsumes(edm::ConsumesCollector&) final;
  void getEventContent(const edm::EventBase&) final;

  double value(const reco::CandidatePtr& cand) const final;

  CandidateType candidateType() const final {
    return ELECTRON;
  }

private:

  // Cut values
  const std::vector<double> mvaCutValues;

  // Pre-computed MVA value map
  edm::Handle<edm::ValueMap<float> > mvaValueMap_;
  edm::Handle<edm::ValueMap<int> > mvaCategoriesMap_;

};

DEFINE_EDM_PLUGIN(CutApplicatorFactory,
                  GsfEleMVAExpoScalingCut,
                  "GsfEleMVAExpoScalingCut");

GsfEleMVAExpoScalingCut::GsfEleMVAExpoScalingCut(const edm::ParameterSet& c) :
  CutApplicatorWithEventContentBase(c),
  mvaCutValues(c.getParameter<std::vector<double> >("mvaCuts"))
{
  edm::InputTag mvaValTag = c.getParameter<edm::InputTag>("mvaValueMapName");
  contentTags_.emplace("mvaVal",mvaValTag);

  edm::InputTag mvaCatTag = c.getParameter<edm::InputTag>("mvaCategoriesMapName");
  contentTags_.emplace("mvaCat",mvaCatTag);

}

void GsfEleMVAExpoScalingCut::setConsumes(edm::ConsumesCollector& cc) {

  auto mvaVal =
    cc.consumes<edm::ValueMap<float> >(contentTags_["mvaVal"]);
  contentTokens_.emplace("mvaVal",mvaVal);

  auto mvaCat =
    cc.consumes<edm::ValueMap<int> >(contentTags_["mvaCat"]);
  contentTokens_.emplace("mvaCat",mvaCat);
}

void GsfEleMVAExpoScalingCut::getEventContent(const edm::EventBase& ev) {

  ev.getByLabel(contentTags_["mvaVal"], mvaValueMap_);
  ev.getByLabel(contentTags_["mvaCat"], mvaCategoriesMap_);
}

CutApplicatorBase::result_type
GsfEleMVAExpoScalingCut::
operator()(const reco::GsfElectronPtr& cand) const{

  // in case we are by-value
  const std::string& val_name = contentTags_.find("mvaVal")->second.instance();
  const std::string& cat_name = contentTags_.find("mvaCat")->second.instance();
  edm::Ptr<pat::Electron> pat(cand);
  float val = -1.0;
  int   cat = -1;
  if( mvaCategoriesMap_.isValid() && mvaCategoriesMap_->contains( cand.id() ) &&
      mvaValueMap_.isValid() && mvaValueMap_->contains( cand.id() ) ) {
    cat = (*mvaCategoriesMap_)[cand];
    val = (*mvaValueMap_)[cand];
  } else if ( mvaCategoriesMap_.isValid() && mvaValueMap_.isValid() &&
              mvaCategoriesMap_->idSize() == 1 && mvaValueMap_->idSize() == 1 &&
              cand.id() == edm::ProductID() ) {
    // in case we have spoofed a ptr
    //note this must be a 1:1 valuemap (only one product input)
    cat = mvaCategoriesMap_->begin()[cand.key()];
    val = mvaValueMap_->begin()[cand.key()];
  } else if ( mvaCategoriesMap_.isValid() && mvaValueMap_.isValid() ){ // throw an exception
    cat = (*mvaCategoriesMap_)[cand];
    val = (*mvaValueMap_)[cand];
  }

  // Find the cut value
  const int iCategory = mvaCategoriesMap_.isValid() ? cat : pat->userInt( cat_name );
  if( iCategory >= (int)(mvaCutValues.size()) * 3)
    throw cms::Exception(" Error in MVA categories: ")
      << " found a particle with a category larger than max configured " << std::endl;
  const float c   = mvaCutValues[3*iCategory];
  const float tau = mvaCutValues[3*iCategory+1];
  const float A   = mvaCutValues[3*iCategory+2];

  // Look up the MVA value for this particle
  const float mvaValue = mvaValueMap_.isValid() ? val : pat->userFloat( val_name );

  // Apply the cut and return the result
  return mvaValue > c - exp(- cand->pt() / tau) * A;;
}

double GsfEleMVAExpoScalingCut::value(const reco::CandidatePtr& cand) const {

  // in case we are by-value
  const std::string& val_name =contentTags_.find("mvaVal")->second.instance();
  edm::Ptr<pat::Electron> pat(cand);
  float val = 0.0;
  if( mvaCategoriesMap_.isValid() && mvaCategoriesMap_->contains( cand.id() ) &&
      mvaValueMap_.isValid() && mvaValueMap_->contains( cand.id() ) ) {
    val = (*mvaValueMap_)[cand];
  } else if ( mvaCategoriesMap_.isValid() && mvaValueMap_.isValid() &&
              mvaCategoriesMap_->idSize() == 1 && mvaValueMap_->idSize() == 1 &&
              cand.id() == edm::ProductID() ) {
    // in case we have spoofed a ptr
    //note this must be a 1:1 valuemap (only one product input)
    val = mvaValueMap_->begin()[cand.key()];
  } else if ( mvaCategoriesMap_.isValid() && mvaValueMap_.isValid() ){ // throw an exception
    val = (*mvaValueMap_)[cand];
  }

  const float mvaValue = mvaValueMap_.isValid() ? val : pat->userFloat( val_name );
  return mvaValue;
}
