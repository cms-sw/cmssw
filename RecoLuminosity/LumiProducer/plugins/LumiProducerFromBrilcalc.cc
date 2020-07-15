// -*- C++ -*-
//
// Package:    RecoLuminosity/LumiProducer
// Class:      LumiProducerFromBrilcalc
//
/**\class LumiProducerFromBrilcalc LumiProducerFromBrilcalc.cc RecoLuminosity/LumiProducer/plugins/LumiProducerFromBrilcalc.cc

   Description: Takes a csv file with luminosity information produced by brilcalc and reads it into the event.

   Implementation:
   [Notes on implementation]
*/
//
// Original Author:  Paul Lujan
//         Created:  Fri, 28 Feb 2020 16:32:38 GMT
//
//

// system include files
#include <memory>
#include <sstream>
#include <fstream>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Luminosity/interface/LumiInfo.h"

//
// class declaration
//

class LumiProducerFromBrilcalc : public edm::global::EDProducer<> {
public:
  explicit LumiProducerFromBrilcalc(const edm::ParameterSet&);
  ~LumiProducerFromBrilcalc() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  const std::string lumiFile_;
  const bool throwIfNotFound_;
  const bool doBunchByBunch_;
  std::map<std::pair<int, int>, std::pair<float, float> > lumiData_;
};

//
// constructor
//

LumiProducerFromBrilcalc::LumiProducerFromBrilcalc(const edm::ParameterSet& iConfig)
    : lumiFile_(iConfig.getParameter<std::string>("lumiFile")),
      throwIfNotFound_(iConfig.getParameter<bool>("throwIfNotFound")),
      doBunchByBunch_(iConfig.getParameter<bool>("doBunchByBunch")) {
  //register your products
  produces<LumiInfo>("brilcalc");

  //now do what ever other initialization is needed
  if (doBunchByBunch_) {
    throw cms::Exception("LumiProducerFromBrilcalc")
        << "Sorry, bunch-by-bunch luminosity is not yet supported. Please bug your friendly lumi expert!";
  }

  // Read luminosity data and store it in lumiData_.
  edm::LogInfo("LumiProducerFromBrilcalc") << "Reading luminosity data from " << lumiFile_ << "...one moment...";
  std::ifstream lumiFile(lumiFile_);
  if (!lumiFile.is_open()) {
    throw cms::Exception("LumiProducerFromBrilcalc") << "Failed to open input luminosity file " << lumiFile_;
  }

  int nLS = 0;
  std::string line;
  while (true) {
    std::getline(lumiFile, line);
    if (lumiFile.eof() || lumiFile.fail())
      break;
    if (line.empty())
      continue;  // skip blank lines
    if (line.at(0) == '#')
      continue;  // skip comment lines

    // Break into fields. These should be, in order: run:fill, brills:cmsls, time, beam status, beam energy,
    // delivered lumi, recorded lumi, average pileup, source.
    std::stringstream ss(line);
    std::string field;
    std::vector<std::string> fields;

    while (std::getline(ss, field, ','))
      fields.push_back(field);

    if (fields.size() != 9) {
      edm::LogWarning("LumiProducerFromBrilcalc") << "Malformed line in csv file: " << line;
      continue;
    }

    // Extract the run number from fields[0] and the lumisection number from fields[1]. Fortunately since the
    // thing we want is before the colon, we don't have to split them again.
    int run, ls;
    std::stringstream runfill(fields[0]);
    runfill >> run;
    std::stringstream lsls(fields[1]);
    lsls >> ls;

    // Extract the delivered and recorded lumi from fields[5] and fields[6].
    float lumiDel, lumiRec, dtFrac;
    std::stringstream lumiDelString(fields[5]);
    lumiDelString >> lumiDel;
    std::stringstream lumiRecString(fields[6]);
    lumiRecString >> lumiRec;

    // Calculate the deadtime fraction
    dtFrac = 1.0 - lumiRec / lumiDel;

    // Finally, store it all.
    lumiData_[std::make_pair(run, ls)] = std::make_pair(lumiDel, dtFrac);
    nLS++;
  }
  edm::LogInfo("LumiProducerFromBrilcalc") << "Read " << nLS << " lumisections from " << lumiFile_;
  lumiFile.close();
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void LumiProducerFromBrilcalc::produce(edm::StreamID iStreamID,
                                       edm::Event& iEvent,
                                       const edm::EventSetup& iSetup) const {
  std::vector<float> bxlumi(3564, 0);
  std::pair<int, int> runls = std::make_pair(iEvent.run(), iEvent.luminosityBlock());
  if (lumiData_.count(runls) == 1) {
    // if we have data for this run/LS, put it in the event
    LogDebug("LumiProducerFromBrilcalc") << "Filling for run " << runls.first << " ls " << runls.second
                                         << " with delivered " << lumiData_.at(runls).first << " dt "
                                         << lumiData_.at(runls).second;
    iEvent.put(std::make_unique<LumiInfo>(lumiData_.at(runls).second, bxlumi, lumiData_.at(runls).first), "brilcalc");
  } else {
    if (throwIfNotFound_) {
      throw cms::Exception("LumiProducerFromBrilcalc")
          << "Failed to find luminosity for run " << runls.first << " LS " << runls.second;
    } else {
      // just put in zeroes
      edm::LogWarning("LumiProducerFromBrilcalc")
          << "Failed to find luminosity for run " << runls.first << " ls " << runls.second;
      iEvent.put(std::make_unique<LumiInfo>(0, bxlumi, 0), "brilcalc");
    }
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void LumiProducerFromBrilcalc::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // Allowed parameters
  edm::ParameterSetDescription desc;
  desc.add<std::string>("lumiFile");
  desc.add<bool>("throwIfNotFound", false);
  desc.add<bool>("doBunchByBunch", false);
  descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(LumiProducerFromBrilcalc);
