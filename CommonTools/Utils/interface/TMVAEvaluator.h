#ifndef CommonTools_Utils_TMVAEvaluator_h
#define CommonTools_Utils_TMVAEvaluator_h

#include <memory>
#include <string>
#include <vector>
#include <map>

#include "TMVA/Reader.h"


class TMVAEvaluator {

  public:
    TMVAEvaluator();
    ~TMVAEvaluator();

    void initialize(const std::string & options, const std::string & method, const std::string & weightFile,
                    const std::vector<std::string> & variables, const std::vector<std::string> & spectators);
    float evaluate(const std::map<std::string,float> & inputs);

  private:
    bool mIsInitialized;

    std::string mMethod;
    std::unique_ptr<TMVA::Reader> mReader;

    std::map<std::string,float> mVariables;
    std::map<std::string,float> mSpectators;
};

#endif // CommonTools_Utils_TMVAEvaluator_h

