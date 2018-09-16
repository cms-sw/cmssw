// * Author: Alberto Zucchetta
// * Mail: a.zucchetta@cern.ch
// * January 16, 2015

#ifndef RecoBTag_SoftLepton_MuonTagger_h
#define RecoBTag_SoftLepton_MuonTagger_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/MVAUtils/interface/TMVAEvaluator.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"
#include <memory>

class MuonTagger : public JetTagComputer {

  public:
  
    MuonTagger(const edm::ParameterSet&);
    void initialize(const JetTagComputerRecord &) override;
    float discriminator(const TagInfoHelper& tagInfo) const override;
    
  private:
    btag::LeptonSelector m_selector;
    const bool m_useCondDB;
    const std::string m_gbrForestLabel;
    const edm::FileInPath m_weightFile;
    const bool m_useGBRForest;
    const bool m_useAdaBoost;

    std::unique_ptr<TMVAEvaluator> mvaID;
};

#endif

