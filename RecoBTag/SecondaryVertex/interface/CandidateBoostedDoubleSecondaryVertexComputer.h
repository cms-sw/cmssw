#ifndef RecoBTag_SecondaryVertex_CandidateBoostedDoubleSecondaryVertexComputer_h
#define RecoBTag_SecondaryVertex_CandidateBoostedDoubleSecondaryVertexComputer_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/MVAUtils/interface/TMVAEvaluator.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"


class CandidateBoostedDoubleSecondaryVertexComputer : public JetTagComputer {

  public:
    CandidateBoostedDoubleSecondaryVertexComputer(const edm::ParameterSet & parameters);

    void  initialize(const JetTagComputerRecord &) override;
    float discriminator(const TagInfoHelper & tagInfos) const override;

  private:
    const bool useCondDB_;
    const std::string gbrForestLabel_;
    const edm::FileInPath weightFile_;
    const bool useGBRForest_;
    const bool useAdaBoost_;

    std::unique_ptr<TMVAEvaluator> mvaID;
};

#endif // RecoBTag_SecondaryVertex_CandidateBoostedDoubleSecondaryVertexComputer_h
