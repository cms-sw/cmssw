// -*- C++ -*-
//
// Package:    LumiProducer
// Class:      LumiProducer
// 
/**\class LumiProducer LumiProducer.cc RecoLuminosity/LumiProducer/src/LumiProducer.cc

Description: This class would load the luminosity object into a Luminosity Block

Implementation:
The are two main steps, the first one retrieve the record of the luminosity
data from the DB and the second loads the Luminosity Obj into the Lumi Block.
*/
//
// Original Author:  Valerie Halyo
//                   David Dagenhart
//       
//         Created:  Tue Jun 12 00:47:28 CEST 2007
// $Id: LumiProducer.cc,v 1.2 2007/07/24 17:58:25 valerieh Exp $

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Luminosity/interface/LumiSummary.h"
#include "DataFormats/Luminosity/interface/LumiDetails.h"

#include "FWCore/Framework/interface/MakerMacros.h"


#include <sstream>
#include <string>
#include <memory>
#include <vector>

namespace edm {
  class EventSetup;
}

//
// class declaration
//

class LumiProducer : public edm::EDProducer {

  public:

    explicit LumiProducer(const edm::ParameterSet&);
    ~LumiProducer();

  private:

    virtual void produce(edm::Event&, const edm::EventSetup&);

    virtual void beginLuminosityBlock(edm::LuminosityBlock & iLBlock,
                                      edm::EventSetup const& iSetup);
    edm::ParameterSet pset_;
};

//
// constructors and destructor
//
LumiProducer::LumiProducer(const edm::ParameterSet& iConfig)
{
  // register your products
  produces<LumiSummary, edm::InLumi>();
  produces<LumiDetails, edm::InLumi>();

  pset_ = iConfig;
}

LumiProducer::~LumiProducer()
{ }


//
// member functions
//

void LumiProducer::produce(edm::Event&, const edm::EventSetup&)
{ }

void LumiProducer::beginLuminosityBlock(edm::LuminosityBlock &iLBlock, edm::EventSetup const &iSetup) {

  unsigned int lumiNumber = iLBlock.id().luminosityBlock();
  std::stringstream ss;
  ss << "LS" << lumiNumber;
  std::string psetName = ss.str();

  // Some defaults that are used only if the configuration file
  // does not have the necessary parameters, We do not
  // want a crash in this case.
  edm::ParameterSet def;
  std::vector<int> defveci;
  std::vector<double> defvecd;

  edm::ParameterSet lumiBlockPSet =
    pset_.getUntrackedParameter<edm::ParameterSet>(psetName, def);

  std::auto_ptr<LumiSummary> pOut1(new LumiSummary(
                          lumiBlockPSet.getUntrackedParameter<double>("avginslumi", -99.0),
			  lumiBlockPSet.getUntrackedParameter<double>("avginslumierr", -99.0),
			  lumiBlockPSet.getUntrackedParameter<int>("lumisecqual", -1),
			  lumiBlockPSet.getUntrackedParameter<double>("deadfrac", -99.0),
			  lumiBlockPSet.getUntrackedParameter<int>("lsnumber", -1),
			  lumiBlockPSet.getUntrackedParameter<std::vector<int> >("l1ratecounter", defveci),
			  lumiBlockPSet.getUntrackedParameter<std::vector<int> >("l1scaler", defveci),
			  lumiBlockPSet.getUntrackedParameter<std::vector<int> >("hltratecounter", defveci),
			  lumiBlockPSet.getUntrackedParameter<std::vector<int> >("hltscaler", defveci),
			  lumiBlockPSet.getUntrackedParameter<std::vector<int> >("hltinput", defveci) ) );
  iLBlock.put(pOut1);

  std::auto_ptr<LumiDetails> pOut2(new LumiDetails(
                          lumiBlockPSet.getUntrackedParameter<std::vector<double> >("lumietsum", defvecd),
			  lumiBlockPSet.getUntrackedParameter<std::vector<double> >("lumietsumerr", defvecd),
			  lumiBlockPSet.getUntrackedParameter<std::vector<int> >("lumietsumqual", defveci),
			  lumiBlockPSet.getUntrackedParameter<std::vector<double> >("lumiocc", defvecd),
			  lumiBlockPSet.getUntrackedParameter<std::vector<double> >("lumioccerr", defvecd),
			  lumiBlockPSet.getUntrackedParameter<std::vector<int> >("lumioccqual", defveci) ) );
  iLBlock.put(pOut2);

  //  ESHandle<SetupData> pLumiSetup; 
  //  iSetup.get<SetupRecord>().get(pLumiSetup); 
}

DEFINE_FWK_MODULE(LumiProducer);
