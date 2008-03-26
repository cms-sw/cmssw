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
(Actually in the initial implementation it is retrieving from the ParameterSet
from the configuration file, the DB is not implemented yet)
*/
//
// Original Author:  Valerie Halyo
//                   David Dagenhart
//       
//         Created:  Tue Jun 12 00:47:28 CEST 2007
// $Id: LumiProducer.cc,v 1.3 2007/07/24 21:51:29 wdd Exp $

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
                          static_cast<float>(lumiBlockPSet.getUntrackedParameter<double>("avginsdellumi", -99.0)),
			  static_cast<float>(lumiBlockPSet.getUntrackedParameter<double>("avginsdellumierr", -99.0)),
			  lumiBlockPSet.getUntrackedParameter<int>("lumisecqual", -1),
			  static_cast<float>(lumiBlockPSet.getUntrackedParameter<double>("deadfrac", -99.0)),
			  lumiBlockPSet.getUntrackedParameter<int>("lsnumber", -1),
			  lumiBlockPSet.getUntrackedParameter<std::vector<int> >("l1ratecounter", defveci),
			  lumiBlockPSet.getUntrackedParameter<std::vector<int> >("l1scaler", defveci),
			  lumiBlockPSet.getUntrackedParameter<std::vector<int> >("hltratecounter", defveci),
			  lumiBlockPSet.getUntrackedParameter<std::vector<int> >("hltscaler", defveci),
			  lumiBlockPSet.getUntrackedParameter<std::vector<int> >("hltinput", defveci) ) );
  iLBlock.put(pOut1);

  // One minor complication here.  The configuration language allows double or
  // vector<double> but not float or vector<float> so we need to explicitly
  // convert from double to float

  typedef std::vector<double>::const_iterator DIter;

  std::vector<double> temp =
    lumiBlockPSet.getUntrackedParameter<std::vector<double> >("lumietsum", defvecd);
  std::vector<float> lumietsum;
  for (DIter i = temp.begin(), e = temp.end(); i != e; ++i) {
    lumietsum.push_back(static_cast<float>(*i));
  }

  temp =
    lumiBlockPSet.getUntrackedParameter<std::vector<double> >("lumietsumerr", defvecd);
  std::vector<float> lumietsumerr;
  for (DIter i = temp.begin(), e = temp.end(); i != e; ++i) {
    lumietsumerr.push_back(static_cast<float>(*i));
  }

  temp = lumiBlockPSet.getUntrackedParameter<std::vector<double> >("lumiocc", defvecd);
  std::vector<float> lumiocc;
  for (DIter i = temp.begin(), e = temp.end(); i != e; ++i) {
    lumiocc.push_back(static_cast<float>(*i));
  }

  temp = lumiBlockPSet.getUntrackedParameter<std::vector<double> >("lumioccerr", defvecd);
  std::vector<float> lumioccerr;
  for (DIter i = temp.begin(), e = temp.end(); i != e; ++i) {
    lumioccerr.push_back(static_cast<float>(*i));
  }

  std::auto_ptr<LumiDetails> pOut2(new LumiDetails(
                          lumietsum,
			  lumietsumerr,
			  lumiBlockPSet.getUntrackedParameter<std::vector<int> >("lumietsumqual", defveci),
			  lumiocc,
			  lumioccerr,
			  lumiBlockPSet.getUntrackedParameter<std::vector<int> >("lumioccqual", defveci) ) );
  iLBlock.put(pOut2);

  //  ESHandle<SetupData> pLumiSetup; 
  //  iSetup.get<SetupRecord>().get(pLumiSetup); 
}

DEFINE_FWK_MODULE(LumiProducer);
