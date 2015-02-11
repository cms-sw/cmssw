// * Author: Alberto Zucchetta
// * Mail: a.zucchetta@cern.ch
// * January 16, 2015

#ifndef RecoBTag_SoftLepton_MuonTagger_h
#define RecoBTag_SoftLepton_MuonTagger_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoBTau/JetTagComputer/interface/JetTagComputer.h"
#include "RecoBTag/SoftLepton/interface/LeptonSelector.h"
#include "RecoBTag/SoftLepton/interface/MvaSoftMuonEstimator.h"

#include "TRandom3.h"

class MuonTagger : public JetTagComputer {

  public:
  
    MuonTagger(const edm::ParameterSet&);
    ~MuonTagger();
    
    virtual float discriminator(const TagInfoHelper& tagInfo) const;
    
  private:
    
    btag::LeptonSelector m_selector;
    TRandom3* random;
    edm::FileInPath WeightFile;
    MvaSoftMuonEstimator* mvaID;
};

#endif

