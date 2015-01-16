// * Author: Alberto Zucchetta
// * Mail: a.zucchetta@cern.ch
// * January 16, 2015

#ifndef RecoBTag_SoftLepton_MvaSoftMuonEstimator_h
#define RecoBTag_SoftLepton_MvaSoftMuonEstimator_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/BTauReco/interface/SoftLeptonTagInfo.h"

#include <TROOT.h>
#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"

class MvaSoftMuonEstimator {

  public:
  
    MvaSoftMuonEstimator();
    ~MvaSoftMuonEstimator();
    
   float mvaValue(float, float, float, float);
    
  private:
    
    TMVA::Reader* TMVAReader;
    
    std::string weightFile;
    float mva_sip3d, mva_sip2d, mva_ptRel, mva_ratio;

};

#endif

