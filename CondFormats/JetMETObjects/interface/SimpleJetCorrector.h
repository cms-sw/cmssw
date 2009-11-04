#ifndef SimpleJetCorrector_h
#define SimpleJetCorrector_h

#include <string>
#include "TFormula.h"

class JetCorrectorParameters;

class SimpleJetCorrector {
 public:
  //-------- Constructors --------------
  SimpleJetCorrector();
  SimpleJetCorrector(const std::string& fDataFile, const std::string& fOption = "");
  //-------- Destructor -----------------
  ~SimpleJetCorrector();
  //-------- Member functions -----------
  void   setInterpolation(bool fInterpolation) {mDoInterpolation = fInterpolation;}
  double correction(const std::vector<float>& fX,const std::vector<float>& fY) const;
  const  JetCorrectorParameters& parameters() const {return *mParameters;} 

 protected:
  //-------- Member functions -----------
  SimpleJetCorrector(const SimpleJetCorrector&);
  SimpleJetCorrector& operator= (const SimpleJetCorrector&);
  double correctionBin(unsigned fBin,const std::vector<float>& fY) const;
  double quadraticInterpolation(double fZ, const double fX[3], const double fY[3]) const; 
  //-------- Member variables -----------
  bool                    mDoInterpolation; 
  TFormula*               mFunc;
  JetCorrectorParameters* mParameters;
};

#endif


