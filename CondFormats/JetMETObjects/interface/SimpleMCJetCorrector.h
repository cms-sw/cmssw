//
// Original Author:  Fedor Ratnikov Dec 27, 2006
// $Id: SimpleMCJetCorrector.h,v 1.5 2008/05/20 23:38:36 fedor Exp $
//
// MC Jet Corrector
//
#ifndef SimpleMCJetCorrector_h
#define SimpleMCJetCorrector_h

#include <map>
#include <string>
namespace MCJetNameSpace{
  class ParametrizationMCJet;
  typedef std::map <double, ParametrizationMCJet*> ParametersMap;
}

class SimpleMCJetCorrector {
 public:
  SimpleMCJetCorrector ();
  SimpleMCJetCorrector (const std::string& fDataFile);
  virtual ~SimpleMCJetCorrector ();

  void init (const std::string& fDataFile);

  virtual double correctionXYZT (double fPx, double fPy, double fPz, double fE) const;
  virtual double correctionPtEtaPhiE (double fPt, double fEta, double fPhi, double fE) const;
  virtual double correctionEtEta (double fEt, double fEta) const;

 private:
  SimpleMCJetCorrector (const SimpleMCJetCorrector&);
  SimpleMCJetCorrector& operator= (const SimpleMCJetCorrector&);
  MCJetNameSpace::ParametersMap* mParametrization;
};

#endif
