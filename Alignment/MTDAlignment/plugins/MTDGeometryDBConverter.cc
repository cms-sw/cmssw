// -*- C++ -*-
//
// Package:    MTDGeometryDBConverter
// Class:      MTDGeometryDBConverter
//
/**\class MTDGeometryDBConverter MTDGeometryDBConverter.cc Alignment/MTDAlignment/plugins/MTDGeometryDBConverter.cc

 Description: <one line class summary>
*/
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
#include "Alignment/MTDAlignment/interface/MTDAlignment.h"
#include "Alignment/MTDAlignment/interface/MTDAlignmentInputMethod.h"
#include "Alignment/MTDAlignment/interface/MTDAlignmentInputDB.h"
#include "Alignment/MTDAlignment/interface/MTDAlignmentInputSurveyDB.h"
#include "Alignment/MTDAlignment/interface/MTDAlignmentInputXML.h"
#include "Alignment/MTDAlignment/interface/MuonScenarioBuilder.h"
#include "Geometry/Records/interface/MTDGeometryRecord.h"
#include "CondFormats/AlignmentRecord/interface/GlobalPositionRcd.h"

//
// class decleration
//

class MTDGeometryDBConverter : public edm::one::EDAnalyzer<> {
public:
  explicit MTDGeometryDBConverter(const edm::ParameterSet &);
  ~MTDGeometryDBConverter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &);
  void beginJob() override {}
  void endJob() override {}

private:
  void analyze(const edm::Event &, const edm::EventSetup &) override;

  bool m_done;
  std::string m_input, m_output;

  std::string m_mtdLabel, m_mtdAPELabel;
  double m_shiftErr, m_angleErr;
  std::string m_fileName;
  bool m_getAPEs;

  edm::ParameterSet m_misalignmentScenario;
  edm::ParameterSet m_outputXML;
  const std::string idealGeometryLabelForInputXML, idealGeometryLabel;

  const edm::ESGetToken<MTDGeometry, MTDGeometryRecord> mtdGeomIdealToken_;
  const edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscGeomIdealToken_;
  const edm::ESGetToken<GEMGeometry, MuonGeometryRecord> gemGeomIdealToken_;

  edm::ESGetToken<MTDGeometry, MTDGeometryRecord> mtdGeomToken_;

  edm::ESGetToken<Alignments, MTDAlignmentRcd> mtdAliToken_;

  edm::ESGetToken<AlignmentErrorsExtended, MTDAlignmentErrorExtendedRcd> mtdAPEToken_;

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
MTDGeometryDBConverter::MTDGeometryDBConverter(const edm::ParameterSet &iConfig)
    : m_done(false),
      m_input(iConfig.getParameter<std::string>("input")),
      m_output(iConfig.getParameter<std::string>("output")),
      m_shiftErr(0.),
      m_angleErr(0.),
      m_getAPEs(false),
      idealGeometryLabelForInputXML("idealForInputXML"),
      idealGeometryLabel("idealGeometry"),
      mtdGeomIdealToken_(esConsumes(edm::ESInputTag("", idealGeometryLabel))),
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
    m_mtdLabel = iConfig.getParameter<std::string>("mtdLabel");
    m_mtdAPELabel = iConfig.getParameter<std::string>("mtdAPELabel");
    m_shiftErr = iConfig.getParameter<double>("shiftErr");
    m_angleErr = iConfig.getParameter<double>("angleErr");
    m_getAPEs = iConfig.getParameter<bool>("getAPEs");
    m_outputXML = iConfig.getParameter<edm::ParameterSet>("outputXML");

    mtdAliToken_ = esConsumes(edm::ESInputTag("", m_mtdLabel));

    mtdAPEToken_ = esConsumes(edm::ESInputTag("", m_mtdAPELabel));

    mtdGeomToken_ = esConsumes(edm::ESInputTag("", idealGeometryLabelForInputXML));
  } else if (m_input == std::string("scenario")) {
    m_misalignmentScenario = iConfig.getParameter<edm::ParameterSet>("MisalignmentScenario");
    m_shiftErr = iConfig.getParameter<double>("shiftErr");
    m_angleErr = iConfig.getParameter<double>("angleErr");
  } else if (m_input == std::string("xml")) {
    m_fileName = iConfig.getParameter<std::string>("fileName");
    m_shiftErr = iConfig.getParameter<double>("shiftErr");
    m_angleErr = iConfig.getParameter<double>("angleErr");
    mtdGeomToken_ = esConsumes(edm::ESInputTag("", idealGeometryLabelForInputXML));
  } else {
    throw cms::Exception("BadConfig") << "input must be \"ideal\", \"db\", or \"xml\"." << std::endl;
  }
  if (m_output == std::string("none")) {
  } else if (m_output == std::string("db")) {
  } else if (m_output == std::string("xml")) {
    m_outputXML = iConfig.getParameter<edm::ParameterSet>("outputXML");
  } else {
    throw cms::Exception("BadConfig") << "output must be \"none\", \"db \", \"xml\"." << std::endl;
  }
}

MTDGeometryDBConverter::~MTDGeometryDBConverter() {}

// ------------ method called to for each event  ------------
void MTDGeometryDBConverter::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  if (!m_done) {
    if (m_input == std::string("ideal")) {
      MTDAlignmentInputMethod inputMethod(&iSetup.getData(mtdGeomIdealToken_), &iSetup.getData(gemGeomIdealToken_));
      MTDAlignment *mtdAlignment = new MTDAlignment(iSetup, inputMethod);
      mtdAlignment->saveToDB();
    } else if (m_input == std::string("db")) {
      MTDAlignmentInputDB inputMethod(&iSetup.getData(mtdGeomIdealToken_),
                                      &iSetup.getData(mtdAliToken_),
                                      &iSetup.getData(mtdAPEToken_),
                                      &iSetup.getData(gprToken_));
      MTDAlignment *mtdAlignment = new MTDAlignment(iSetup, inputMethod);
      mtdAlignment->writeXML(m_outputXML, &iSetup.getData(mtdGeomToken_));
    } else if (m_input == std::string("scenario")) {
      MTDAlignmentInputMethod inputMethod(&iSetup.getData(mtdGeomIdealToken_));
      MTDAlignment *mtdAlignment = new MTDAlignment(iSetup, inputMethod);

      MTDScenarioBuilder mtdScenarioBuilder(mtdAlignment->getAlignableMuon());
      mtdScenarioBuilder.applyScenario(m_misalignmentScenario);
      mtdAlignment->saveToDB();
    } else if (m_input == std::string("xml")) {
      MTDAlignmentInputXML inputMethod(m_fileName, &iSetup.getData(mtdGeomToken_), &iSetup.getData(mtdGeomIdealToken_));
      MTDAlignment *mtdAlignment = new MTDAlignment(iSetup, inputMethod);
      mtdAlignment->saveToDB();
      mtdAlignment->fillGapsInSurvey(m_shiftErr, m_angleErr);
    }
    m_done = true;
  }  // end if not done
  else {
    throw cms::Exception("BadConfig") << "Set maxEvents.input to 1.  (Your output is okay.)" << std::endl;
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void MTDGeometryDBConverter::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Converts mtd geometry between various formats.");
  desc.add<std::string>("input", "ideal");
  desc.add<std::string>("mtdLabel", "");
  desc.add<std::string>("mtdAPELabel", "");
  desc.add<double>("shiftErr", 1000.0);
  desc.add<double>("angleErr", 6.28);
  desc.add<bool>("getAPEs", true);
  desc.add<std::string>("output", "xml");
  desc.add<std::string>("fileName", "REPLACEME.xml");
  edm::ParameterSetDescription outputXML;
  outputXML.add<std::string>("fileName", "REPLACEME.xml");
  outputXML.add<std::string>("relativeto", "ideal");
  outputXML.add<bool>("rawIds", false);
  outputXML.add<bool>("eulerAngles", false);
  outputXML.add<int>("precision", 10);
  desc.add("outputXML", outputXML);
  descriptions.add("muonGeometryDBConverter", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(MTDGeometryDBConverter);
