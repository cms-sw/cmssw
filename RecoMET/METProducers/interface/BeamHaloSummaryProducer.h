#ifndef RECOMET_METPRODUCERS_BEAMHALOSUMMARYPRODUCER_H
#define RECOMET_METPRODUCERS_BEAMHALOSUMMARYPRODUCER_H

/*
  [class]:  BeamHaloSummaryProducer
  [authors]: R. Remington, The University of Florida
  [description]: EDProducer which runs BeamHalo Id/Flagging algorithms and stores BeamHaloSummary object to the event. Inspiration for this implementation was taken from HcalNoisInfoProducer.cc by J.P Chou
  [date]: October 15, 2009
*/  


//Standard C++ classes
#include <iostream>
#include <string>
#include <map>
#include <vector>
#include <utility>
#include <ostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <memory>
#include <iomanip>
#include <cstdlib>

// user include files
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/METReco/interface/CSCHaloData.h"
#include "DataFormats/METReco/interface/EcalHaloData.h"
#include "DataFormats/METReco/interface/HcalHaloData.h"
#include "DataFormats/METReco/interface/GlobalHaloData.h"
#include "DataFormats/METReco/interface/BeamHaloSummary.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

namespace reco
{
  class BeamHaloSummaryProducer : public edm::stream::EDProducer<> {
    
  public:
    explicit BeamHaloSummaryProducer(const edm::ParameterSet&);
    ~BeamHaloSummaryProducer();
    
  private:
    
    virtual void produce(edm::Event&, const edm::EventSetup&) override;
    
    edm::InputTag IT_CSCHaloData;
    edm::InputTag IT_EcalHaloData;
    edm::InputTag IT_HcalHaloData;
    edm::InputTag IT_GlobalHaloData;

    edm::EDGetTokenT<CSCHaloData> cschalodata_token_;
    edm::EDGetTokenT<EcalHaloData> ecalhalodata_token_;
    edm::EDGetTokenT<HcalHaloData> hcalhalodata_token_;
    edm::EDGetTokenT<GlobalHaloData> globalhalodata_token_;

    float L_EcalPhiWedgeEnergy;
    int L_EcalPhiWedgeConstituents;
    float L_EcalPhiWedgeToF;
    float L_EcalPhiWedgeConfidence;
    float L_EcalShowerShapesRoundness;
    float L_EcalShowerShapesAngle;
    int L_EcalSuperClusterSize;
    float L_EcalSuperClusterEnergy;

    float T_EcalPhiWedgeEnergy;
    int T_EcalPhiWedgeConstituents;
    float T_EcalPhiWedgeToF;
    float T_EcalPhiWedgeConfidence;
    float T_EcalShowerShapesRoundness;
    float T_EcalShowerShapesAngle;
    int T_EcalSuperClusterSize;
    float T_EcalSuperClusterEnergy;
    
    float L_HcalPhiWedgeEnergy;
    int L_HcalPhiWedgeConstituents;
    float L_HcalPhiWedgeToF;
    float L_HcalPhiWedgeConfidence;
    
    float T_HcalPhiWedgeEnergy;
    int T_HcalPhiWedgeConstituents;
    float T_HcalPhiWedgeToF;
    float T_HcalPhiWedgeConfidence;
  };
}

#endif
  
