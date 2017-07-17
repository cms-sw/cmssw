#include "RecoEgamma/EgammaTools/interface/MVAObjectCache.h"

using namespace egamma;

MVAObjectCache::MVAObjectCache(const edm::ParameterSet& conf) {
  const std::vector<edm::ParameterSet>& mvaEstimatorConfigs
    = conf.getParameterSetVector("mvaConfigurations");
  
  for( auto &imva : mvaEstimatorConfigs ){
    // building the mva class is now done in the ObjectCache, 
    // so we loop over what's in that.
    std::unique_ptr<AnyMVAEstimatorRun2Base> thisEstimator;
    thisEstimator.reset(nullptr);
    if( !imva.empty() ) {
      const std::string& pName = imva.getParameter<std::string>("mvaName");
      // The factory below constructs the MVA of the appropriate type based
      // on the "mvaName" which is the name of the derived MVA class (plugin)      
      const AnyMVAEstimatorRun2Base *estimator = AnyMVAEstimatorRun2Factory::get()->create( pName, imva );
      // Declare all event content, such as ValueMaps produced upstream or other,
      // original event data pieces, that is needed (if any is implemented in the specific
      // MVA classes)
      const std::string full_name = estimator->getName() + estimator->getTag();
      auto diditwork = mvas_.emplace( full_name, MVAPtr(estimator) );
      if( !diditwork.second ) {
        throw cms::Exception("MVA configured twice: ")
          <<  "Tried already to make an mva of name: " << estimator->getName()
          << " please ensure that the name of the MVA is unique!" << std::endl;
      }
    } else {
      throw cms::Exception(" MVA configuration not found: ")
	<< " failed to find proper configuration for "
        <<"one of the MVAs in the main python script " << std::endl;
    }
  }
}

const MVAObjectCache::MVAPtr& 
MVAObjectCache::getMVA(const std::string& mva) const {
  auto itr = mvas_.find(mva);
  if( itr == mvas_.end() ) {
    throw cms::Exception("InvalidMVAName")
      << mva << " is not managed by this evaluator!";
  }
  return itr->second;
}
