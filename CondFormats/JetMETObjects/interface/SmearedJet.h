#ifndef SmearedJet_h
#define SmearedJet_h

#include <string>
#include <vector>
#include "TF1.h"
#include "TRandom.h"
#include "Math/PtEtaPhiE4D.h"
#include "Math/LorentzVector.h"

typedef ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiE4D<double> > PtEtaPhiELorentzVectorD;
class SimpleJetCorrectorParameters;

class SmearedJet {
 public:
  enum ParType {pResp = 0, pPtResol, pEtaResol, pPhiResol, pRecoEff};
  enum VarType {vRawPt = 0, vPt, vEta, vPhi};
  //------- default constructor ------- 
  SmearedJet();

  //------- destructor -------
  virtual ~SmearedJet ();

  //------- parameter setter -------
  void setParameters (const std::string& fDataFile, ParType fOption);

  //------- returns the Pt,Eta,Phi resolutions or the reco efficiency -------
  double getValue (double fPt, double fEta, ParType fOption);

  //------- returns a smeared value -------
  double getSmeared (double fPt, double fEta, double fPhi, VarType fOption);

  //------- returns a smeared vector -------
  PtEtaPhiELorentzVectorD getSmeared (PtEtaPhiELorentzVectorD fP4, VarType fOption); 

  //------- returns true if reconstructed ------
  bool isReconstructed(double fPt, double fEta);
  bool isReconstructed(PtEtaPhiELorentzVectorD fP4);
  
 private:
  ///// ----- member functions -------
  
  //------- copy method -------
  SmearedJet (const SmearedJet&);

  //------- equal sign operator -------
  SmearedJet& operator= (const SmearedJet&);

  //------- option checkers -------
  void checkOption(ParType fOption);
  void checkOption(VarType fOption);  

  ///// ----- member variables -------
  TRandom*                                  mRnd;
  std::vector<TF1>                          mFunction;
  std::vector<bool>                         mIsSet;
  std::vector<std::string>                  mFormula;
  std::vector<std::string>                  mName;
  std::vector<SimpleJetCorrectorParameters> mParameters;
};

#endif


