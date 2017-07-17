#ifndef __ElectronIdentification_SoftElectronMVAEstimator_H__
#define __ElectronIdentification_SoftElectronMVAEstimator_H__

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include <memory>
#include <string>

class SoftElectronMVAEstimator {
 public:
  constexpr static unsigned int ExpectedNBins = 1;

  struct Configuration{
    std::vector<std::string> vweightsfiles;
  };
  SoftElectronMVAEstimator(const Configuration & );
  ~SoftElectronMVAEstimator() ;
  double mva(const reco::GsfElectron& myElectron,
             const reco::VertexCollection&) const;
  UInt_t   GetMVABin(int pu,double eta,double pt ) const;
 private:
  void bindVariables(float vars[25]) const ;
  void init();

 private:
    const Configuration cfg_;
    std::array<std::unique_ptr<const GBRForest>, ExpectedNBins> gbr;
    
    Float_t                    fbrem;
    Float_t                    EtotOvePin;
    Float_t                    EBremOverDeltaP;
    Float_t                    logSigmaEtaEta;
    Float_t                    DeltaEtaTrackEcalSeed;
    Float_t                    kfchi2;
    Float_t                    kfhits;    //number of layers
    Float_t                    gsfchi2;
    Float_t                    SigmaPtOverPt;


    Float_t                    deta;
    Float_t                    dphi;
    Float_t                    detacalo;

    Float_t                    see;
    Float_t                    etawidth;
    Float_t                    phiwidth;
    Float_t                    OneMinusE1x5E5x5;

    Float_t                    HoE;
    //Float_t                    EoP; //Not being used
    Float_t                    eleEoPout;

    Float_t 			spp;
    Float_t 			R9;
    Float_t			IoEmIoP;
    Float_t			PreShowerOverRaw;                             


    Float_t                    eta;
    Float_t                    pt;  
   
    Float_t 			nPV;
};

#endif
