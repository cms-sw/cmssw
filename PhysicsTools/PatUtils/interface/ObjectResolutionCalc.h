//
//

#ifndef PhysicsTools_PatUtils_ObjectResolutionCalc_h
#define PhysicsTools_PatUtils_ObjectResolutionCalc_h

/**
  \class    pat::ObjectResolutionCalc ObjectResolutionCalc.h "PhysicsTools/PatUtils/interface/ObjectResolutionCalc.h"
  \brief    Class to calculate MC resolutions for pat objects

  \author   Jan Heyninck, Petra Van Mulders, Christophe Delaere
  \version  $Id: ObjectResolutionCalc.h,v 1.5 2008/10/08 19:19:25 gpetrucc Exp $
*/

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

#include "TF1.h"
#include "TFile.h"
#include "TMultiLayerPerceptron.h"
#include "TString.h"

namespace pat {

  class ObjectResolutionCalc {
  public:
    ObjectResolutionCalc();
    ObjectResolutionCalc(const TString& resopath, bool useNN);
    ~ObjectResolutionCalc();

    float obsRes(int obs, int eta, float eT);
    int etaBin(float eta);

#ifdef OBSOLETE
    void operator()(Electron& obj);
    void operator()(Muon& obj);
    void operator()(Tau& obj);
    void operator()(Jet& obj);
    void operator()(MET& obj);
#else
    // WORKAROUND
    template <typename T>
    void operator()(T& obj) {}
#endif

  private:
    TFile* resoFile_;
    std::vector<float> etaBinVals_;
    TF1 fResVsEt_[10][10];
    TMultiLayerPerceptron* network_[10];
    bool useNN_;
  };

}  // namespace pat

#endif
