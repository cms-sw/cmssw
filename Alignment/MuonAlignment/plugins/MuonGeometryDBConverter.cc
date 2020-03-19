// -*- C++ -*-
//
// Package:    MuonGeometryDBConverter
// Class:      MuonGeometryDBConverter
//
/**\class MuonGeometryDBConverter MuonGeometryDBConverter.cc Alignment/MuonAlignment/plugins/MuonGeometryDBConverter.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Pivarski
//         Created:  Sat Feb 16 00:04:55 CST 2008
// $Id: MuonGeometryDBConverter.cc,v 1.15 2011/09/15 09:12:01 mussgill Exp $
//
//

// system include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// user include files
#include "Alignment/MuonAlignment/interface/MuonAlignment.h"
#include "Alignment/MuonAlignment/interface/MuonAlignmentInputMethod.h"
#include "Alignment/MuonAlignment/interface/MuonAlignmentInputDB.h"
#include "Alignment/MuonAlignment/interface/MuonAlignmentInputSurveyDB.h"
#include "Alignment/MuonAlignment/interface/MuonAlignmentInputXML.h"
#include "Alignment/MuonAlignment/interface/MuonScenarioBuilder.h"

//
// class decleration
//

class MuonGeometryDBConverter : public edm::one::EDAnalyzer<> {
public:
  explicit MuonGeometryDBConverter(const edm::ParameterSet &);
  ~MuonGeometryDBConverter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &);
  void beginJob() override{};
  void endJob() override{};

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  bool m_done;
  std::string m_input, m_output;

  std::string m_dtLabel, m_cscLabel;
  double m_shiftErr, m_angleErr;
  std::string m_fileName;
  bool m_getAPEs;

  edm::ParameterSet m_misalignmentScenario;
  edm::ParameterSet m_outputXML;
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
MuonGeometryDBConverter::MuonGeometryDBConverter(const edm::ParameterSet &iConfig)
    : m_done(false),
      m_input(iConfig.getParameter<std::string>("input")),
      m_output(iConfig.getParameter<std::string>("output")),
      m_shiftErr(0.),
      m_angleErr(0.),
      m_getAPEs(false) {
  ////////////////////////////////////////////////////////////////////
  // Version V02-03-02 and earlier of this module had support for   //
  // "cfg" as an input/output format.  It turns out that reading    //
  // thousands of parameters from a configuration file takes a very //
  // long time, so "cfg" wasn't very practical.  When I reorganized //
  // the code, I didn't bother to port it.                          //
  ////////////////////////////////////////////////////////////////////

  if (m_input == std::string("ideal")) {
  }

  else if (m_input == std::string("db")) {
    m_dtLabel = iConfig.getParameter<std::string>("dtLabel");
    m_cscLabel = iConfig.getParameter<std::string>("cscLabel");
    m_shiftErr = iConfig.getParameter<double>("shiftErr");
    m_angleErr = iConfig.getParameter<double>("angleErr");
    m_getAPEs = iConfig.getParameter<bool>("getAPEs");
  }

  else if (m_input == std::string("surveydb")) {
    m_dtLabel = iConfig.getParameter<std::string>("dtLabel");
    m_cscLabel = iConfig.getParameter<std::string>("cscLabel");
  }

  else if (m_input == std::string("scenario")) {
    m_misalignmentScenario = iConfig.getParameter<edm::ParameterSet>("MisalignmentScenario");
    m_shiftErr = iConfig.getParameter<double>("shiftErr");
    m_angleErr = iConfig.getParameter<double>("angleErr");
  }

  else if (m_input == std::string("xml")) {
    m_fileName = iConfig.getParameter<std::string>("fileName");
    m_shiftErr = iConfig.getParameter<double>("shiftErr");
    m_angleErr = iConfig.getParameter<double>("angleErr");
  }

  else {
    throw cms::Exception("BadConfig") << "input must be \"ideal\", \"db\", \"surveydb\", or \"xml\"." << std::endl;
  }

  if (m_output == std::string("none")) {
  }

  else if (m_output == std::string("db")) {
  }

  else if (m_output == std::string("surveydb")) {
  }

  else if (m_output == std::string("xml")) {
    m_outputXML = iConfig.getParameter<edm::ParameterSet>("outputXML");
  }

  else {
    throw cms::Exception("BadConfig") << "output must be \"none\", \"db\", or \"surveydb\"." << std::endl;
  }
}

MuonGeometryDBConverter::~MuonGeometryDBConverter() {}

// ------------ method called to for each event  ------------
void MuonGeometryDBConverter::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  if (!m_done) {
    MuonAlignment *muonAlignment = nullptr;

    if (m_input == std::string("ideal")) {
      MuonAlignmentInputMethod inputMethod;
      muonAlignment = new MuonAlignment(iSetup, inputMethod);
      muonAlignment->fillGapsInSurvey(0., 0.);
    }

    else if (m_input == std::string("db")) {
      MuonAlignmentInputDB inputMethod(m_dtLabel, m_cscLabel, m_getAPEs);
      muonAlignment = new MuonAlignment(iSetup, inputMethod);
      if (m_getAPEs) {
        muonAlignment->copyAlignmentToSurvey(m_shiftErr, m_angleErr);
      }
    }

    else if (m_input == std::string("surveydb")) {
      MuonAlignmentInputSurveyDB inputMethod(m_dtLabel, m_cscLabel);
      muonAlignment = new MuonAlignment(iSetup, inputMethod);
      muonAlignment->copySurveyToAlignment();
    }

    else if (m_input == std::string("scenario")) {
      MuonAlignmentInputMethod inputMethod;
      muonAlignment = new MuonAlignment(iSetup, inputMethod);

      MuonScenarioBuilder muonScenarioBuilder(muonAlignment->getAlignableMuon());
      muonScenarioBuilder.applyScenario(m_misalignmentScenario);
      muonAlignment->copyAlignmentToSurvey(m_shiftErr, m_angleErr);
    }

    else if (m_input == std::string("xml")) {
      MuonAlignmentInputXML inputMethod(m_fileName);
      muonAlignment = new MuonAlignment(iSetup, inputMethod);
      muonAlignment->fillGapsInSurvey(m_shiftErr, m_angleErr);
    }

    /////////////

    if (muonAlignment) {
      if (m_output == std::string("none")) {
      }

      else if (m_output == std::string("db")) {
        muonAlignment->saveToDB();
      }

      else if (m_output == std::string("surveydb")) {
        muonAlignment->saveSurveyToDB();
      }

      else if (m_output == std::string("xml")) {
        muonAlignment->writeXML(m_outputXML, iSetup);
      }

      delete muonAlignment;
    }

    m_done = true;
  }  // end if not done
  else {
    throw cms::Exception("BadConfig") << "Set maxEvents.input to 1.  (Your output is okay.)" << std::endl;
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void MuonGeometryDBConverter::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Converts muon geometry between various formats.");
  desc.add<std::string>("input", "ideal");
  desc.add<std::string>("dtLabel", "");
  desc.add<std::string>("cscLabel", "");
  desc.add<double>("shiftErr", 1000.0);
  desc.add<double>("angleErr", 6.28);
  desc.add<bool>("getAPEs", true);
  desc.add<std::string>("output", "xml");
  desc.add<std::string>("fileName", "REPLACEME.xml");
  edm::ParameterSetDescription outputXML;
  outputXML.add<std::string>("fileName", "REPLACEME.xml");
  outputXML.add<std::string>("relativeto", "ideal");
  outputXML.add<bool>("survey", false);
  outputXML.add<bool>("rawIds", false);
  outputXML.add<bool>("eulerAngles", false);
  outputXML.add<int>("precision", 10);
  outputXML.addUntracked<bool>("suppressDTBarrel", true);
  outputXML.addUntracked<bool>("suppressDTWheels", true);
  outputXML.addUntracked<bool>("suppressDTStations", true);
  outputXML.addUntracked<bool>("suppressDTChambers", false);
  outputXML.addUntracked<bool>("suppressDTSuperLayers", false);
  outputXML.addUntracked<bool>("suppressDTLayers", false);
  outputXML.addUntracked<bool>("suppressCSCEndcaps", true);
  outputXML.addUntracked<bool>("suppressCSCStations", true);
  outputXML.addUntracked<bool>("suppressCSCRings", true);
  outputXML.addUntracked<bool>("suppressCSCChambers", false);
  outputXML.addUntracked<bool>("suppressCSCLayers", false);
  desc.add("outputXML", outputXML);
  descriptions.add("muonGeometryDBConverter", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonGeometryDBConverter);
