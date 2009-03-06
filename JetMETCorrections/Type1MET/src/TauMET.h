#ifndef Type1MET_TauMET_h
#define Type1MET_TauMET_h

// -*- C++ -*-
//
// Package:    TauMET
// Class:      TauMET
// 
// class TauMET TauMET.cc TauMET.cc
//
// Original Author:  Chi Nhan Nguyen
//         Created:  Mon Oct 22 15:20:51 CDT 2007
// $Id$
//
//

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

using namespace std;
namespace cms 
{
  class TauMET : public edm::EDProducer {
  public:
    explicit TauMET(const edm::ParameterSet&);
    ~TauMET();
    
  private:
    virtual void beginJob(const edm::EventSetup&) ;
    virtual void produce(edm::Event&, const edm::EventSetup&);
    virtual void endJob() ;
    
    // Input parameters
    string _InputPFJetsLabel;
    string _InputCaloJetsLabel;
    string _correctorLabel;
    bool _UseCorrectedJets;
    double _JetMatchDeltaR;

    //std::string _InputTyp1MetLabel;

    // Not used: for tau tagging
    //std::string _InputJetTagLabel;
    //double _TauJetDiscrMin;
    //double _TauEtMin;
    //double _TauAbsEtaMax;

    TauMETAlgo _algo;


  };
}

#endif
