// * Author: Alberto Zucchetta
// * Mail: a.zucchetta@cern.ch
// * January 16, 2015

#ifndef RecoBTag_SoftLepton_MuonTagger_h
#define RecoBTag_SoftLepton_MuonTagger_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/Utils/interface/TMVAEvaluator.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"
#include <mutex>
#include <memory>

class MuonTagger : public JetTagComputer {

  public:
  
    MuonTagger(const edm::ParameterSet&);
    
    virtual float discriminator(const TagInfoHelper& tagInfo) const override;
    
  private:
    
    btag::LeptonSelector m_selector;
    mutable std::mutex m_mutex;
    [[cms::thread_guard("m_mutex")]] std::unique_ptr<TMVAEvaluator> mvaID;
};

#endif

