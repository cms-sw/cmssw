#ifndef RecoBTag_SecondaryVertex_CandidateBoostedDoubleSecondaryVertexComputer_h
#define RecoBTag_SecondaryVertex_CandidateBoostedDoubleSecondaryVertexComputer_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"


class CandidateBoostedDoubleSecondaryVertexComputer : public JetTagComputer {
  public:
    CandidateBoostedDoubleSecondaryVertexComputer(const edm::ParameterSet & parameters);

    float discriminator(const TagInfoHelper & tagInfos) const override;

  private:
    edm::FileInPath tmvaWeightFile_;
};

#endif // RecoBTag_SecondaryVertex_CandidateBoostedDoubleSecondaryVertexComputer_h
