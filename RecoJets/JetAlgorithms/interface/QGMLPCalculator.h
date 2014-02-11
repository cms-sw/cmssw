#ifndef QGMLPCalculator_h
#define QGMLPCalculator_h

#include <TROOT.h>
#include <TMVA/Reader.h>

//#include "FWCore/Framework/interface/Frameworkfwd.h"
//#include "FWCore/Framework/interface/EDProducer.h"
//
//#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/MakerMacros.h"
//#include "FWCore/ParameterSet/interface/FileInPath.h"
//#include "FWCore/ParameterSet/interface/ParameterSet.h"

class QGMLPCalculator{
   public:
      explicit QGMLPCalculator(const TString, const TString, const Bool_t);
      ~QGMLPCalculator(){};
      Float_t QGvalue(std::map<TString, Float_t>);

   private:
      Float_t QGvalueInBin(std::map<TString, Float_t>, TString, Int_t);
      Float_t interpolate(Double_t, Int_t, Int_t, Float_t&, Float_t&);
      TString str(Int_t i){return TString::Format("%d",i);};
      void setRhoCorrections();
      Int_t getNbins();
      Int_t getLastBin(TString);
      Double_t getMinPt();
      Double_t getMaxPt(TString);
      Double_t getBins(Int_t);
      Double_t getBinsAveragePt(TString, Int_t);

      // ----------member data -------------------------
      TString mva;
      Bool_t useProbValue;
      TMVA::Reader *reader;
      std::map<TString, Float_t> corrections, mvaVariables_corr;
};
#endif
