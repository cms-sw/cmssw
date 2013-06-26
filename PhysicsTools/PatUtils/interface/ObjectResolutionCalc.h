//
// $Id: ObjectResolutionCalc.h,v 1.6 2013/02/23 10:14:19 eulisse Exp $
//

#ifndef PhysicsTools_PatUtils_ObjectResolutionCalc_h
#define PhysicsTools_PatUtils_ObjectResolutionCalc_h

/**
  \class    pat::ObjectResolutionCalc ObjectResolutionCalc.h "PhysicsTools/PatUtils/interface/ObjectResolutionCalc.h"
  \brief    Class to calculate MC resolutions for pat objects

  \author   Jan Heyninck, Petra Van Mulders, Christophe Delaere
  \version  $Id: ObjectResolutionCalc.h,v 1.6 2013/02/23 10:14:19 eulisse Exp $
*/


#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

#include "TF1.h"
#include "TH1.h"
#include "TFile.h"
#include "TKey.h"
#include "TString.h"
#include "TMultiLayerPerceptron.h"


namespace pat {


  class ObjectResolutionCalc {

    public:

      ObjectResolutionCalc();
      ObjectResolutionCalc(TString resopath, bool useNN);
      ~ObjectResolutionCalc();

      float obsRes(int obs, int eta, float eT);
      int   etaBin(float eta);

#ifdef OBSOLETE
      void  operator()(Electron & obj);
      void  operator()(Muon & obj);
      void  operator()(Tau & obj);
      void  operator()(Jet & obj);
      void  operator()(MET & obj);
#else
      // WORKAROUND
      template<typename T> void operator()(T &obj) { }
#endif

    private:

      TFile * resoFile_;
      std::vector<float> etaBinVals_;
      TF1 fResVsEt_[10][10];
      TMultiLayerPerceptron * network_[10];
      bool useNN_;

  };


}

#endif
