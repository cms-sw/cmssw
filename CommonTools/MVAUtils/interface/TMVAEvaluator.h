#ifndef CommonTools_MVAUtils_TMVAEvaluator_h
#define CommonTools_MVAUtils_TMVAEvaluator_h

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "CondFormats/GBRForest/interface/GBRForest.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#include "TMVA/IMethod.h"
#include "TMVA/Reader.h"

class TMVAEvaluator {
public:
  TMVAEvaluator();

  void initialize(const std::string& options,
                  const std::string& method,
                  const std::string& weightFile,
                  const std::vector<std::string>& variables,
                  const std::vector<std::string>& spectators,
                  bool useGBRForest = false,
                  bool useAdaBoost = false);

  void initializeGBRForest(const GBRForest* gbrForest,
                           const std::vector<std::string>& variables,
                           const std::vector<std::string>& spectators,
                           bool useAdaBoost = false);

  void initializeGBRForest(const edm::EventSetup& iSetup,
                           const std::string& label,
                           const std::vector<std::string>& variables,
                           const std::vector<std::string>& spectators,
                           bool useAdaBoost = false);

  float evaluateTMVA(const std::map<std::string, float>& inputs, bool useSpectators) const;
  float evaluateGBRForest(const std::map<std::string, float>& inputs) const;
  float evaluate(const std::map<std::string, float>& inputs, bool useSpectators = false) const;

private:
  bool mIsInitialized;
  bool mUsingGBRForest;
  bool mUseAdaBoost;

  std::string mMethod;
  mutable std::mutex m_mutex;
  CMS_THREAD_GUARD(m_mutex) std::unique_ptr<TMVA::Reader> mReader;
  std::shared_ptr<const GBRForest> mGBRForest;

  CMS_THREAD_GUARD(m_mutex) mutable std::map<std::string, std::pair<size_t, float>> mVariables;
  CMS_THREAD_GUARD(m_mutex) mutable std::map<std::string, std::pair<size_t, float>> mSpectators;
};

#endif  // CommonTools_Utils_TMVAEvaluator_h
