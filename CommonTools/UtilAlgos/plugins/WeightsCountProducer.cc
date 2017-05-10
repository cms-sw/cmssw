// -*- C++ -*-
//
// Package:    WeightsCountProducer
// Class:      WeightsCountProducer
// 
/**\class WeightsCountProducer WeightsCountProducer.cc CommonTools/UtilAlgos/plugins/WeightsCountProducer.cc

Description: An event counter that can store the number of events in the lumi block 

*/


// system include files
#include <memory>
#include <vector>
#include <algorithm>
#include <iostream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/Common/interface/MergeableDouble.h"
#include "DataFormats/Common/interface/MergeableHisto.h"


class WeightsCountProducer : public edm::one::EDProducer<edm::one::WatchLuminosityBlocks,
                                                       edm::EndLuminosityBlockProducer> {
public:
  explicit WeightsCountProducer(const edm::ParameterSet&);
  ~WeightsCountProducer();

private:
  virtual void produce(edm::Event &, const edm::EventSetup&) override;
  virtual void beginLuminosityBlock(const edm::LuminosityBlock &, const edm::EventSetup&) override;
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, const edm::EventSetup&) override;
  virtual void endLuminosityBlockProduce(edm::LuminosityBlock &, const edm::EventSetup&) override;
      
  // ----------member data ---------------------------
  /// GenEventInfoProduct_generator__SIM.
  edm::EDGetTokenT<GenEventInfoProduct> genInfoToken_;

  bool doTruePileup_;
  bool doObsPileup_;
  edm::EDGetTokenT<std::vector<PileupSummaryInfo> >   puInfoToken_;
  
  int nbinsTruePileup_;
  double minTruePileup_, maxTruePileup_, widthTruePileup_;
  int nbinsObsPileup_;
  double minObsPileup_, maxObsPileup_, widthObsPileup_;

  double weightProcessedInLumi_;
  typedef edm::MergeableHistoF hysto_type;
  hysto_type::container_type truePileup_, obsPileup_, zeroTruePileup_, zeroObsPileup_;
  
};



using namespace edm;
using namespace std;



WeightsCountProducer::WeightsCountProducer(const edm::ParameterSet& iConfig) :
  genInfoToken_(consumes<GenEventInfoProduct>(iConfig.getParameter<InputTag> ("generator")))
{
  produces<edm::MergeableDouble, edm::InLumi>("totalWeight");
  
  doTruePileup_  = iConfig.getUntrackedParameter<bool>("doTruePileup",false);
  doObsPileup_   = iConfig.getUntrackedParameter<bool>("doObsPileup",false);
  if( doTruePileup_ || doObsPileup_ ) {
    puInfoToken_ = consumes<std::vector<PileupSummaryInfo> >(iConfig.getParameter<InputTag> ("pileupInfo"));
  }
  
  if( doTruePileup_ ) {
    nbinsTruePileup_ = iConfig.getParameter<int>("nbinsTruePileup");
    minTruePileup_ =  iConfig.getParameter<double>("minTruePileup");
    maxTruePileup_ =  iConfig.getParameter<double>("maxTruePileup");
    widthTruePileup_ = (maxTruePileup_-minTruePileup_)/(double)nbinsTruePileup_;
    zeroTruePileup_.resize(nbinsTruePileup_+2,0.);  // add bins for overflow and underdflow
    produces<hysto_type, edm::InLumi>("truePileup");
  }

  if( doObsPileup_ ) {
    nbinsObsPileup_ = iConfig.getParameter<int>("nbinsObsPileup");
    minObsPileup_ =  iConfig.getParameter<double>("minObsPileup");
    maxObsPileup_ =  iConfig.getParameter<double>("maxObsPileup");
    widthObsPileup_ = (maxObsPileup_-minObsPileup_)/(double)nbinsObsPileup_;
    zeroObsPileup_.resize(nbinsObsPileup_+2,0.); // add bins for overflow and underdflow
    produces<hysto_type, edm::InLumi>("obsPileup");
  }

}


WeightsCountProducer::~WeightsCountProducer(){}


void
WeightsCountProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){
  edm::Handle<GenEventInfoProduct> genInfo;
  edm::Handle<std::vector<PileupSummaryInfo> > puInfo;

  if( ! iEvent.isRealData() ) {
    iEvent.getByToken(genInfoToken_,genInfo);
    
    const auto & weights = genInfo->weights(); 
    if( ! weights.empty() ) {
      weightProcessedInLumi_ += weights[0];
      
      if( doTruePileup_ || doObsPileup_ ) {
	iEvent.getByToken(puInfoToken_,puInfo);
	hysto_type::value_type truePu=0., obsPu=0.;
	for( auto & frame : *puInfo ) {
	  /// std::cout << frame.getBunchCrossing() << std::endl;
	  if( frame.getBunchCrossing() == 0 ) {
	    truePu = frame.getTrueNumInteractions();
	    obsPu = frame.getPU_NumInteractions();
	    break;
	  }
	}
	/// cout << truePu << " " << obsPu << endl; 
      
	if( doTruePileup_ ) {
	  size_t bin = 0;
	  if( truePu >= maxTruePileup_ ) { bin = truePileup_.size() - 1; }
	  else if( truePu >= minTruePileup_ ) { bin = (size_t)std::floor( (truePu-minTruePileup_) / widthTruePileup_) + 1; }
	  truePileup_[bin] += weights[0];
	}
	
	if( doObsPileup_ ) {
	  size_t bin = 0;
	  if( obsPu >= maxObsPileup_ ) { bin = obsPileup_.size() - 1; }
	  else if( obsPu >= minObsPileup_ ) { bin = (size_t)std::floor( (obsPu-minObsPileup_) / widthObsPileup_) + 1; }
	  /// cout << bin << " " << std::floor( (obsPu-minObsPileup_) / widthObsPileup_) << endl;
	  obsPileup_[bin] += weights[0];
	}
      }
      
    }
  }  
  
  return;
}


void 
WeightsCountProducer::beginLuminosityBlock(const LuminosityBlock & theLuminosityBlock, const EventSetup & theSetup) {
  weightProcessedInLumi_ = 0.;
  if( doTruePileup_ ) {
    truePileup_ = zeroTruePileup_;
  }
  if( doObsPileup_ ) {
    obsPileup_ = zeroObsPileup_;
  }
  
  return;
}

void 
WeightsCountProducer::endLuminosityBlock(LuminosityBlock const& theLuminosityBlock, const EventSetup & theSetup) {
}

void 
WeightsCountProducer::endLuminosityBlockProduce(LuminosityBlock & theLuminosityBlock, const EventSetup & theSetup) {
  LogTrace("WeightsCounting") << "endLumi: adding " << weightProcessedInLumi_ << " events" << endl;

  auto_ptr<edm::MergeableDouble> numWeightssPtr(new edm::MergeableDouble);
  numWeightssPtr->value = weightProcessedInLumi_;
  theLuminosityBlock.put(numWeightssPtr,"totalWeight");
  
  if( doTruePileup_ ) {
    auto_ptr<hysto_type> truePileup(new hysto_type);
    truePileup->min = minTruePileup_;
    truePileup->max = maxTruePileup_;
    truePileup->values = truePileup_;
    theLuminosityBlock.put(truePileup,"truePileup");
  }
  if( doObsPileup_ ) {
    auto_ptr<hysto_type> obsPileup(new hysto_type);
    obsPileup->min = minObsPileup_;
    obsPileup->max = maxObsPileup_;
    obsPileup->values = obsPileup_;
    theLuminosityBlock.put(obsPileup,"obsPileup");
  }
  
  return;
}



//define this as a plug-in
DEFINE_FWK_MODULE(WeightsCountProducer);
