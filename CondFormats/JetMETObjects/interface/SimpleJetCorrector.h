#ifndef SimpleJetCorrector_h
#define SimpleJetCorrector_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <string>
#include <vector>

#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"

#include "CommonTools/Utils/interface/FormulaEvaluator.h"

class JetCorrectorParameters;

class SimpleJetCorrector 
{
 public:
  //-------- Constructors --------------
  SimpleJetCorrector(const std::string& fDataFile, const std::string& fOption = "");
  SimpleJetCorrector(const JetCorrectorParameters& fParameters);
  //-------- Member functions -----------
  void   setInterpolation(bool fInterpolation) {mDoInterpolation = fInterpolation;}
  float  correction(const std::vector<float>& fX,const std::vector<float>& fY) const;  
  const  JetCorrectorParameters& parameters() const {return mParameters;} 

 private:
  //-------- Member functions -----------
  SimpleJetCorrector(const SimpleJetCorrector&);
  SimpleJetCorrector& operator= (const SimpleJetCorrector&);
  float    invert(const double *args, const double *params) const;
  float    correctionBin(unsigned fBin,const std::vector<float>& fY) const;
  unsigned findInvertVar();
  void     setFuncParameters();
  //-------- Member variables -----------
  JetCorrectorParameters  mParameters;
  reco::FormulaEvaluator  mFunc;
  unsigned                mInvertVar; 
  bool                    mDoInterpolation;
};

#endif


