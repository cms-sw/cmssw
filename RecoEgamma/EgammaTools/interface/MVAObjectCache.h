#ifndef __RecoEgamma_EgammaTools_MVAObjectCache_H__
#define __RecoEgamma_EgammaTools_MVAObjectCache_H__

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoEgamma/EgammaTools/interface/AnyMVAEstimatorRun2Base.h"

#include <memory>
#include <string>
#include <unordered_map>

namespace egamma {

  class MVAObjectCache {    
  public:
    typedef std::unique_ptr<const AnyMVAEstimatorRun2Base> MVAPtr;
    
    MVAObjectCache(const edm::ParameterSet& conf);

    const MVAPtr& getMVA(const std::string& mva) const;

    const std::unordered_map<std::string,MVAPtr>& allMVAs() const {
      return mvas_;
    }
  private:
    std::unordered_map<std::string,MVAPtr> mvas_;   
  };

}
#endif
