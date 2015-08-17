#ifndef CommonTools_Utils_TMVAEvaluator_h
#define CommonTools_Utils_TMVAEvaluator_h

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <mutex>

#include "TMVA/Reader.h"
#include "TMVA/IMethod.h"
#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include "FWCore/Framework/interface/EventSetup.h"


class TMVAEvaluator {

  public:
    TMVAEvaluator();
    ~TMVAEvaluator();

    void initialize(const std::string & options, const std::string & method, const std::string & weightFile,
                    const std::vector<std::string> & variables, const std::vector<std::string> & spectators, bool useGBRForest=false, bool useAdaBoost=false);
    void initializeGBRForest(const GBRForest* gbrForest, const std::vector<std::string> & variables,
                             const std::vector<std::string> & spectators, bool useAdaBoost=false);
    void initializeGBRForest(const edm::EventSetup &iSetup, const std::string & label,
                             const std::vector<std::string> & variables, const std::vector<std::string> & spectators, bool useAdaBoost=false);
    float evaluateTMVA(const std::map<std::string,float> & inputs, bool useSpectators) const;
    float evaluateGBRForest(const std::map<std::string,float> & inputs) const;
    float evaluate(const std::map<std::string,float> & inputs, bool useSpectators=false) const;

  private:
    bool mIsInitialized;
    bool mUsingGBRForest;
    bool mUseAdaBoost;
    bool mReleaseAtEnd;

    std::string mMethod;
    mutable std::mutex m_mutex;
    [[cms::thread_guard("m_mutex")]] std::unique_ptr<TMVA::Reader> mReader;
    std::unique_ptr<TMVA::IMethod> mIMethod;
    std::unique_ptr<const GBRForest> mGBRForest;

    [[cms::thread_guard("m_mutex")]] mutable std::map<std::string,std::pair<size_t,float>> mVariables;
    [[cms::thread_guard("m_mutex")]] mutable std::map<std::string,std::pair<size_t,float>> mSpectators;
};

#endif // CommonTools_Utils_TMVAEvaluator_h

