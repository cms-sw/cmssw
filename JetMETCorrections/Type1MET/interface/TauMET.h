#ifndef Type1MET_TauMET_h
#define Type1MET_TauMET_h

// Original Authors:  Alfredo Gurrola, Chi Nhan Nguyen

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"

#include "JetMETCorrections/Type1MET/src/TauMETAlgo.h"


namespace cms 
{
  class TauMET : public edm::EDProducer {
  public:
    explicit TauMET(const edm::ParameterSet&);
    ~TauMET();
    
  private:
    virtual void beginJob() ;
    virtual void produce(edm::Event&, const edm::EventSetup&);
    virtual void endJob() ;
    
    // Input parameters
    std::string _InputTausLabel;
    std::string _tauType;
    std::string _InputCaloJetsLabel;
    double _jetPTthreshold;
    double _jetEMfracLimit;
    std::string _correctorLabel;
    std::string _InputMETLabel;
    std::string _metType;
    double _JetMatchDeltaR;
    double _TauMinEt;
    double _TauEtaMax;
    bool _UseSeedTrack;
    double _seedTrackPt;
    bool _UseTrackIsolation;
    double _trackIsolationMinPt;
    bool _UseECALIsolation;
    double _gammaIsolationMinPt;
    bool _UseProngStructure;

    TauMETAlgo _algo;


  };
}

#endif
