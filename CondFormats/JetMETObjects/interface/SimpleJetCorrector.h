#ifndef SimpleJetCorrector_h
#define SimpleJetCorrector_h

#include <string>
#include <vector>

#include <TFormula.h>


class JetCorrectorParameters;

class SimpleJetCorrector 
{
 public:
  //-------- Constructors --------------
  SimpleJetCorrector();
  SimpleJetCorrector(const std::string& fDataFile, const std::string& fOption = "");
  SimpleJetCorrector(const JetCorrectorParameters& fParameters);
  //-------- Destructor -----------------
  ~SimpleJetCorrector();
  //-------- Member functions -----------
  void   setInterpolation(bool fInterpolation) {mDoInterpolation = fInterpolation;}
  float  correction(const std::vector<float>& fX,const std::vector<float>& fY) const;  
  const  JetCorrectorParameters& parameters() const {return *mParameters;} 

 private:
  //-------- Member functions -----------
  SimpleJetCorrector(const SimpleJetCorrector&);
  SimpleJetCorrector& operator= (const SimpleJetCorrector&);
  float    invert(std::vector<float> fX) const;
  float    correctionBin(unsigned fBin,const std::vector<float>& fY) const;
  unsigned findInvertVar();
  //-------- Member variables -----------
  bool                    mDoInterpolation;
  unsigned                mInvertVar; 
  TFormula*               mFunc;
  JetCorrectorParameters* mParameters;
};

#endif


