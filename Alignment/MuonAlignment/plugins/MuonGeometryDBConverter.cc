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
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"

//
// class decleration
//

class MuonGeometryDBConverter : public edm::one::EDAnalyzer<> {
public:
  explicit MuonGeometryDBConverter(const edm::ParameterSet &);
  ~MuonGeometryDBConverter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &);
  void beginJob() override {}
  void endJob() override {}

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  bool m_done;
  std::string m_input, m_output;

  std::string m_dtLabel, m_cscLabel, m_gemLabel, m_dtAPELabel, m_cscAPELabel, m_gemAPELabel;
  double m_shiftErr, m_angleErr;
  std::string m_fileName;
  bool m_getAPEs;

  edm::ParameterSet m_misalignmentScenario;
  edm::ParameterSet m_outputXML;
  const std::string idealGeometryLabelForInputXML, idealGeometryLabel;

  const edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomIdealToken_;
  const edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscGeomIdealToken_;
  const edm::ESGetToken<GEMGeometry, MuonGeometryRecord> gemGeomIdealToken_;

  edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeomToken_;
  edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscGeomToken_;
  edm::ESGetToken<GEMGeometry, MuonGeometryRecord> gemGeomToken_;

  edm::ESGetToken<Alignments, DTAlignmentRcd> dtAliToken_;
  edm::ESGetToken<Alignments, CSCAlignmentRcd> cscAliToken_;
  edm::ESGetToken<Alignments, GEMAlignmentRcd> gemAliToken_;

  edm::ESGetToken<AlignmentErrorsExtended, DTAlignmentErrorExtendedRcd> dtAPEToken_;
  edm::ESGetToken<AlignmentErrorsExtended, CSCAlignmentErrorExtendedRcd> cscAPEToken_;
  edm::ESGetToken<AlignmentErrorsExtended, GEMAlignmentErrorExtendedRcd> gemAPEToken_;

  const edm::ESGetToken<Alignments, GlobalPositionRcd> gprToken_;
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
      m_getAPEs(false),
      idealGeometryLabelForInputXML("idealForInputXML"),
      idealGeometryLabel("idealGeometry"),
      dtGeomIdealToken_(esConsumes(edm::ESInputTag("", idealGeometryLabel))),
      cscGeomIdealToken_(esConsumes(edm::ESInputTag("", idealGeometryLabel))),
      gemGeomIdealToken_(esConsumes(edm::ESInputTag("", idealGeometryLabel))),
      gprToken_(esConsumes<Alignments, GlobalPositionRcd>(edm::ESInputTag("", ""))) {
  ////////////////////////////////////////////////////////////////////
  // Version V02-03-02 and earlier of this module had support for   //
  // "cfg" as an input/output format.  It turns out that reading    //
  // thousands of parameters from a configuration file takes a very //
  // long time, so "cfg" wasn't very practical.  When I reorganized //
  // the code, I didn't bother to port it.                          //
  ////////////////////////////////////////////////////////////////////

  if (m_input == std::string("ideal")) {
  } else if (m_input == std::string("db")) {
    m_dtLabel = iConfig.getParameter<std::string>("dtLabel");
    m_cscLabel = iConfig.getParameter<std::string>("cscLabel");
    m_gemLabel = iConfig.getParameter<std::string>("gemLabel");
    m_dtAPELabel = iConfig.getParameter<std::string>("dtAPELabel");
    m_cscAPELabel = iConfig.getParameter<std::string>("cscAPELabel");
    m_gemAPELabel = iConfig.getParameter<std::string>("gemAPELabel");
    m_shiftErr = iConfig.getParameter<double>("shiftErr");
    m_angleErr = iConfig.getParameter<double>("angleErr");
    m_getAPEs = iConfig.getParameter<bool>("getAPEs");
    m_outputXML = iConfig.getParameter<edm::ParameterSet>("outputXML");

    dtAliToken_ = esConsumes(edm::ESInputTag("", m_dtLabel));
    cscAliToken_ = esConsumes(edm::ESInputTag("", m_cscLabel));
    gemAliToken_ = esConsumes(edm::ESInputTag("", m_gemLabel));

    dtAPEToken_ = esConsumes(edm::ESInputTag("", m_dtAPELabel));
    cscAPEToken_ = esConsumes(edm::ESInputTag("", m_cscAPELabel));
    gemAPEToken_ = esConsumes(edm::ESInputTag("", m_gemAPELabel));

    dtGeomToken_ = esConsumes(edm::ESInputTag("", idealGeometryLabelForInputXML));
    cscGeomToken_ = esConsumes(edm::ESInputTag("", idealGeometryLabelForInputXML));
    gemGeomToken_ = esConsumes(edm::ESInputTag("", idealGeometryLabelForInputXML));
  } else if (m_input == std::string("surveydb")) {
    m_dtLabel = iConfig.getParameter<std::string>("dtLabel");
    m_cscLabel = iConfig.getParameter<std::string>("cscLabel");
    m_gemLabel = iConfig.getParameter<std::string>("gemLabel");
  } else if (m_input == std::string("scenario")) {
    m_misalignmentScenario = iConfig.getParameter<edm::ParameterSet>("MisalignmentScenario");
    m_shiftErr = iConfig.getParameter<double>("shiftErr");
    m_angleErr = iConfig.getParameter<double>("angleErr");
  } else if (m_input == std::string("xml")) {
    m_fileName = iConfig.getParameter<std::string>("fileName");
    m_shiftErr = iConfig.getParameter<double>("shiftErr");
    m_angleErr = iConfig.getParameter<double>("angleErr");
    dtGeomToken_ = esConsumes(edm::ESInputTag("", idealGeometryLabelForInputXML));
    cscGeomToken_ = esConsumes(edm::ESInputTag("", idealGeometryLabelForInputXML));
    gemGeomToken_ = esConsumes(edm::ESInputTag("", idealGeometryLabelForInputXML));
  } else {
    throw cms::Exception("BadConfig") << "input must be \"ideal\", \"db\", \"surveydb\", or \"xml\"." << std::endl;
  }
  if (m_output == std::string("none")) {
  } else if (m_output == std::string("db")) {
  } else if (m_output == std::string("xml")) {
    m_outputXML = iConfig.getParameter<edm::ParameterSet>("outputXML");
  } else {
    throw cms::Exception("BadConfig") << "output must be \"none\", \"db \", \"xml\"." << std::endl;
  }
}

MuonGeometryDBConverter::~MuonGeometryDBConverter() {}

// ------------ method called to for each event  ------------
void MuonGeometryDBConverter::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  if (!m_done) {
    if (m_input == std::string("ideal")) {
      MuonAlignmentInputMethod inputMethod(
          &iSetup.getData(dtGeomIdealToken_), &iSetup.getData(cscGeomIdealToken_), &iSetup.getData(gemGeomIdealToken_));
      MuonAlignment *muonAlignment = new MuonAlignment(iSetup, inputMethod);
      muonAlignment->fillGapsInSurvey(0., 0.);
      muonAlignment->saveToDB();
    } else if (m_input == std::string("db")) {
      MuonAlignmentInputDB inputMethod(&iSetup.getData(dtGeomIdealToken_),
                                       &iSetup.getData(cscGeomIdealToken_),
                                       &iSetup.getData(gemGeomIdealToken_),
                                       &iSetup.getData(dtAliToken_),
                                       &iSetup.getData(cscAliToken_),
                                       &iSetup.getData(gemAliToken_),
                                       &iSetup.getData(dtAPEToken_),
                                       &iSetup.getData(cscAPEToken_),
                                       &iSetup.getData(gemAPEToken_),
                                       &iSetup.getData(gprToken_));
      MuonAlignment *muonAlignment = new MuonAlignment(iSetup, inputMethod);
      if (m_getAPEs) {
        muonAlignment->copyAlignmentToSurvey(m_shiftErr, m_angleErr);
      }
      muonAlignment->writeXML(
          m_outputXML, &iSetup.getData(dtGeomToken_), &iSetup.getData(cscGeomToken_), &iSetup.getData(gemGeomToken_));
    } else if (m_input == std::string("scenario")) {
      MuonAlignmentInputMethod inputMethod(
          &iSetup.getData(dtGeomIdealToken_), &iSetup.getData(cscGeomIdealToken_), &iSetup.getData(gemGeomIdealToken_));
      MuonAlignment *muonAlignment = new MuonAlignment(iSetup, inputMethod);

      MuonScenarioBuilder muonScenarioBuilder(muonAlignment->getAlignableMuon());
      muonScenarioBuilder.applyScenario(m_misalignmentScenario);
      muonAlignment->saveToDB();
      muonAlignment->copyAlignmentToSurvey(m_shiftErr, m_angleErr);
    } else if (m_input == std::string("xml")) {
      MuonAlignmentInputXML inputMethod(m_fileName,
                                        &iSetup.getData(dtGeomToken_),
                                        &iSetup.getData(cscGeomToken_),
                                        &iSetup.getData(gemGeomToken_),
                                        &iSetup.getData(dtGeomIdealToken_),
                                        &iSetup.getData(cscGeomIdealToken_),
                                        &iSetup.getData(gemGeomIdealToken_));
      MuonAlignment *muonAlignment = new MuonAlignment(iSetup, inputMethod);
      muonAlignment->saveToDB();
      muonAlignment->fillGapsInSurvey(m_shiftErr, m_angleErr);
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
  desc.add<std::string>("gemLabel", "");
  desc.add<std::string>("dtAPELabel", "");
  desc.add<std::string>("cscAPELabel", "");
  desc.add<std::string>("gemAPELabel", "");
  desc.add<double>("shiftErr", 1000.0);
  desc.add<double>("angleErr", 6.28);
  desc.add<bool>("getAPEs", true);
  desc.add<std::string>("output", "xml");
  desc.add<std::string>("fileName", "REPLACEME.xml");
  edm::ParameterSetDescription outputXML;
  outputXML.add<std::string>("fileName", "REPLACEME.xml");
  outputXML.add<std::string>("relativeto", "ideal");
  outputXML.add<bool>("rawIds", false);
  outputXML.add<bool>("survey", false);
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
  outputXML.addUntracked<bool>("suppressGEMEndcaps", true);
  outputXML.addUntracked<bool>("suppressGEMStations", true);
  outputXML.addUntracked<bool>("suppressGEMRings", true);
  outputXML.addUntracked<bool>("suppressGEMSuperChambers", false);
  outputXML.addUntracked<bool>("suppressGEMChambers", true);
  outputXML.addUntracked<bool>("suppressGEMEtaPartitions", true);
  desc.add("outputXML", outputXML);
  descriptions.add("muonGeometryDBConverter", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonGeometryDBConverter);
-- dummy change --
