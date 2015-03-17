// * Author: Alberto Zucchetta
// * Mail: a.zucchetta@cern.ch
// * January 16, 2015

#ifndef RecoBTag_SoftLepton_MuonTagger_h
#define RecoBTag_SoftLepton_MuonTagger_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"
#include "RecoBTag/SoftLepton/interface/MvaSoftMuonEstimator.h"
#include <mutex>
#include <memory>

class MuonTagger : public JetTagComputer {

  public:
  
    MuonTagger(const edm::ParameterSet&);
    
    virtual float discriminator(const TagInfoHelper& tagInfo) const override;
    
  private:
    
    btag::LeptonSelector m_selector;
    edm::FileInPath WeightFile;
    mutable std::mutex m_mutex;
    [[cms::thread_guard("m_mutex")]] std::unique_ptr<MvaSoftMuonEstimator> mvaID;
};

#endif

