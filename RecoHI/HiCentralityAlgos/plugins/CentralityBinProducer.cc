// -*- C++ -*-
//
// Package:    CentralityBinProducer
// Class:      CentralityBinProducer
//
/**\class CentralityBinProducer CentralityBinProducer.cc RecoHI/CentralityBinProducer/src/CentralityBinProducer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Yetkin Yilmaz
//         Created:  Thu Aug 12 05:34:11 EDT 2010
//
//


// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/HeavyIonEvent/interface/Centrality.h"
#include "CondFormats/HIObjects/interface/CentralityTable.h"
#include "CondFormats/DataRecord/interface/HeavyIonRcd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"


//
// class declaration
//

class CentralityBinProducer : public edm::EDProducer {
  enum VariableType {HFtowers = 0, HFtowersPlus = 1, HFtowersMinus = 2, HFtowersTrunc = 3, HFtowersPlusTrunc = 4, HFtowersMinusTrunc = 5, HFhits = 6, PixelHits = 7, PixelTracks = 8, Tracks = 9, EB = 10, EE = 11, ZDChitsPlus = 12, ZDChitsMinus = 13, Missing = 14};
   public:
      explicit CentralityBinProducer(const edm::ParameterSet&);
      ~CentralityBinProducer() override;

   private:
      void beginRun(edm::Run const& run, const edm::EventSetup& iSetup) override;
      void produce(edm::Event&, const edm::EventSetup&) override;

      // ----------member data ---------------------------


  edm::Handle<reco::Centrality> chandle_;
  edm::EDGetTokenT<reco::Centrality> tag_;
  edm::ESHandle<CentralityTable> inputDB_;

  std::string centralityVariable_;
  std::string centralityLabel_;
  std::string centralityMC_;
  unsigned int prevRun_;

  VariableType varType_;
  unsigned int pPbRunFlip_;

};

//
// constants, enums and typedefs
//


//
// static data member definitions
//

//
// constructors and destructor
//
CentralityBinProducer::CentralityBinProducer(const edm::ParameterSet& iConfig):
  prevRun_(0),
  varType_(Missing)
{
   using namespace edm;
   tag_ = consumes<reco::Centrality>(iConfig.getParameter<edm::InputTag>("Centrality"));
   centralityVariable_ = iConfig.getParameter<std::string>("centralityVariable");
   pPbRunFlip_ = iConfig.getParameter<unsigned int>("pPbRunFlip");

   if(centralityVariable_ == "HFtowers") varType_ = HFtowers;
   if(centralityVariable_ == "HFtowersPlus") varType_ = HFtowersPlus;
   if(centralityVariable_ == "HFtowersMinus") varType_ = HFtowersMinus;
   if(centralityVariable_ == "HFtowersTrunc") varType_ = HFtowersTrunc;
   if(centralityVariable_ == "HFtowersPlusTrunc") varType_ = HFtowersPlusTrunc;
   if(centralityVariable_ == "HFtowersMinusTrunc") varType_ = HFtowersMinusTrunc;
   if(centralityVariable_ == "HFhits") varType_ = HFhits;
   if(centralityVariable_ == "PixelHits") varType_ = PixelHits;
   if(centralityVariable_ == "PixelTracks") varType_ = PixelTracks;
   if(centralityVariable_ == "Tracks") varType_ = Tracks;
   if(centralityVariable_ == "EB") varType_ = EB;
   if(centralityVariable_ == "EE") varType_ = EE;
   if(centralityVariable_ == "ZDChitsPlus") varType_ = ZDChitsPlus;
   if(centralityVariable_ == "ZDChitsMinus") varType_ = ZDChitsMinus;
   if(varType_ == Missing){
     std::string errorMessage="Requested Centrality variable does not exist : "+centralityVariable_+"\n" +
       "Supported variables are: \n" + "HFtowers HFtowersPlus HFtowersMinus HFtowersTrunc HFtowersPlusTrunc HFtowersMinusTrunc HFhits PixelHits PixelTracks Tracks EB EE" + "\n";
     throw cms::Exception("Configuration",errorMessage);
   }

   if(iConfig.exists("nonDefaultGlauberModel")){
     centralityMC_ = iConfig.getParameter<std::string>("nonDefaultGlauberModel");
   }
   centralityLabel_ = centralityVariable_+centralityMC_;

   produces<int>(centralityVariable_);
}


CentralityBinProducer::~CentralityBinProducer()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
CentralityBinProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{

  iEvent.getByToken(tag_,chandle_);

  double value = 0;
  switch(varType_){
  case HFtowers : value = chandle_->EtHFtowerSum();break;
  case HFtowersPlus : value = chandle_->EtHFtowerSumPlus();break;
  case HFtowersMinus : value = chandle_->EtHFtowerSumMinus();break;
  case HFhits : value = chandle_->EtHFhitSum();break;
  case HFtowersTrunc : value = chandle_->EtHFtruncated();break;
  case HFtowersPlusTrunc : value = chandle_->EtHFtruncatedPlus();break;
  case HFtowersMinusTrunc : value = chandle_->EtHFtruncatedMinus();break;
  case PixelHits : value = chandle_->multiplicityPixel();break;
  case PixelTracks : value = chandle_->NpixelTracks();break;
  case Tracks : value = chandle_->Ntracks();break;
  case EB : value = chandle_->EtEBSum();break;
  case EE : value = chandle_->EtEESum();break;
  case ZDChitsPlus : value = chandle_->zdcSumPlus();break;
  case ZDChitsMinus : value = chandle_->zdcSumMinus();break;
  default:
    throw cms::Exception("CentralityBinProducer","Centrality variable not recognized.");
  }

  int bin = inputDB_->m_table.size() - 1;
  for(unsigned int i = 0; i < inputDB_->m_table.size(); ++i){

    if(value >= inputDB_->m_table[i].bin_edge && value){
      bin = i; break;
    }

  }

  iEvent.put(std::make_unique<int>(bin),centralityVariable_);

}


void
CentralityBinProducer::beginRun(edm::Run const& iRun, const edm::EventSetup& iSetup)
{

  if(prevRun_ < pPbRunFlip_ && iRun.run() >= pPbRunFlip_){
     if(centralityVariable_ == "HFtowersPlus") varType_ = HFtowersMinus;
     if(centralityVariable_ == "HFtowersMinus") varType_ = HFtowersPlus;
     if(centralityVariable_ == "HFtowersPlusTrunc") varType_ = HFtowersMinusTrunc;
     if(centralityVariable_ == "HFtowersMinusTrunc") varType_ = HFtowersPlusTrunc;
     if(centralityVariable_ == "ZDChitsPlus") varType_ = ZDChitsMinus;
     if(centralityVariable_ == "ZDChitsMinus") varType_ = ZDChitsPlus;
  }
  prevRun_ = iRun.run();

  iSetup.get<HeavyIonRcd>().get(centralityLabel_,inputDB_);

}

//define this as a plug-in
DEFINE_FWK_MODULE(CentralityBinProducer);
