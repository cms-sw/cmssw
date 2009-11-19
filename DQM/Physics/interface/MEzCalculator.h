#ifndef DQM_MEzCalculator_h
#define DQM_MEzCalculator_h

#include <iostream>
#include "TLorentzVector.h"

/**
   \class   MEzCalculator MEzCalculator.h "DQM/Physics/interface/MEzCalculator.h"

   \brief   Add a one sentence description here...

   Add a more detailed description here...
*/

class MEzCalculator {
  
 public:
  /// default constructor
  MEzCalculator();
  /// default destructor
  ~MEzCalculator();
  /// Set MET
  void SetMET(TLorentzVector MET) {
    MET_.SetPxPyPzE(MET.Px(),MET.Py(),MET.Pz(),MET.E());
  }
  /// Set lepton
  void SetLepton(TLorentzVector lepton, bool isMuon = true) {
    lepton_.SetPxPyPzE(lepton.Px(),lepton.Py(),lepton.Pz(),lepton.E());
    isMuon_ = isMuon;
  }
  /// Calculate MEz
  // options to choose roots from quadratic equation:
  // type = 0 : if real roots, pick the one nearest to
  //            the lepton Pz except when the Pz so chosen
  //            is greater than 300 GeV in which case pick
  //            the most central root.
  // type = 1 (default): if real roots, choose the one closest to the lepton Pz
  //           if complex roots, use only the real part.
  // type = 2: if real roots, choose the most central solution.
  //           if complex roots, use only the real part.
  double Calculate(int type = 1);
  /// check for complex root
  bool IsComplex() const { return isComplex_; };
  /// ...
  void Print();

 private:
  /// ...
  bool isComplex_;
  /// ...
  TLorentzVector lepton_;
  /// ...
  TLorentzVector MET_;
  /// ...
  bool isMuon_;
};

#endif
