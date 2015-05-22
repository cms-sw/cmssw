
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "PhysicsTools/Utilities/interface/Lumi3DReWeighting.h"

#include <TH1D.h>
#include <TFile.h>

#include <iostream>
#include <string>

using namespace std;

class PileUpWeight3DProducer : public edm::EDProducer{
 public:
  PileUpWeight3DProducer(const edm::ParameterSet& ps);
  virtual ~PileUpWeight3DProducer();

  virtual void produce(edm::Event&, const edm::EventSetup&);

 private:
  
  edm::Lumi3DReWeighting * LumiWeights_;
  bool verbose_;
};

PileUpWeight3DProducer::PileUpWeight3DProducer(const edm::ParameterSet& ps):
  verbose_(ps.getUntrackedParameter<bool>("verbose",false)) {


  TFile fileData( ps.getParameter<std::string>("inputHistData").c_str() );  
  if(fileData.IsZombie())
    throw cms::Exception("PileUpWeight3DProducer")<<" bad input Data file "<<fileData.GetName();

  TFile fileMC( ps.getParameter<std::string>("inputHistMC").c_str() );
  if(fileMC.IsZombie())
    throw cms::Exception("PileUpWeight3DProducer")<<" bad input MC file "<<fileMC.GetName();
 
  TH1D* histData = (TH1D*)fileData.Get("pileup");
  if(!histData) 
    throw cms::Exception("PileUpWeight3DProducer")<<"Data histogram doesn't exist in file "<<fileData.GetName();

  TH1D* histMC = (TH1D*)fileMC.Get("pileup");
  if(!histMC) 
    throw cms::Exception("PileUpWeight3DProducer")<<"MC histogram doesn't exist in file "<<fileMC.GetName();


  LumiWeights_ = new edm::Lumi3DReWeighting(ps.getParameter<std::string>("inputHistMC").c_str()
					    ,ps.getParameter<std::string>("inputHistData").c_str()
					    , "pileup"
					    , "pileup"
					    , "");
  LumiWeights_->weight3D_init(1.0);//scale factor can be used for systematic variations

  produces<double>();

}



PileUpWeight3DProducer::~PileUpWeight3DProducer() {
  delete  LumiWeights_;
}


void PileUpWeight3DProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){

  //method from https://twiki.cern.ch/twiki/bin/viewauth/CMS/PileupMCReweightingUtilities#3D_Reweighting
  edm::EventBase* iEventB = dynamic_cast<edm::EventBase*>(&iEvent);
  double mcPUPWeight = LumiWeights_->weight3D( (*iEventB) );

  std::auto_ptr<double> output( new double( mcPUPWeight ) ); 
  iEvent.put( output );

}

DEFINE_FWK_MODULE(PileUpWeight3DProducer);

