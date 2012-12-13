/**\class MuonGeometryDBConverter

*/

//
// Original Author:  Jim Pivarski
//         Created:  Sat Feb 16 00:04:55 CST 2008
//
// $Id: MuonGeometryDBConverter.cc,v 1.16 2011/09/15 09:15:51 mussgill Exp $
//

// system include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
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


class MuonGeometryDBConverter : public edm::EDAnalyzer
{
public:

  explicit MuonGeometryDBConverter(const edm::ParameterSet&);

  virtual ~MuonGeometryDBConverter() {};

private:

  virtual void analyze(const edm::Event&, const edm::EventSetup&);

  bool m_done;
  std::string m_input, m_output;

  std::string m_dtLabel, m_cscLabel;
  double m_shiftErr, m_angleErr;
  std::string m_fileName;
  bool m_getAPEs;

  edm::ParameterSet m_misalignmentScenario;
  edm::ParameterSet m_outputXML;
};



MuonGeometryDBConverter::MuonGeometryDBConverter(const edm::ParameterSet &iConfig)
: m_done(false)
, m_input(iConfig.getParameter<std::string>("input"))
, m_output(iConfig.getParameter<std::string>("output"))
, m_shiftErr(0.)
, m_angleErr(0.)
, m_getAPEs(false)
{
  ////////////////////////////////////////////////////////////////////
  // Version V02-03-02 and earlier of this module had support for   //
  // "cfg" as an input/output format.  It turns out that reading    //
  // thousands of parameters from a configuration file takes a very //
  // long time, so "cfg" wasn't very practical.  When I reorganized //
  // the code, I didn't bother to port it.                          //
  ////////////////////////////////////////////////////////////////////

  if (m_input == std::string("ideal")) {}

  else if (m_input == std::string("db"))
  {
    m_dtLabel = iConfig.getParameter<std::string>("dtLabel");
    m_cscLabel = iConfig.getParameter<std::string>("cscLabel");
    m_shiftErr = iConfig.getParameter<double>("shiftErr");
    m_angleErr = iConfig.getParameter<double>("angleErr");
    m_getAPEs = iConfig.getParameter<bool>("getAPEs");
  }
  else if (m_input == std::string("surveydb"))
  {
    m_dtLabel = iConfig.getParameter<std::string>("dtLabel");
    m_cscLabel = iConfig.getParameter<std::string>("cscLabel");
  }
  else if (m_input == std::string("scenario"))
  {
    m_misalignmentScenario = iConfig.getParameter<edm::ParameterSet>("MisalignmentScenario");
    m_shiftErr = iConfig.getParameter<double>("shiftErr");
    m_angleErr = iConfig.getParameter<double>("angleErr");
  }
  else if (m_input == std::string("xml"))
  {
    m_fileName = iConfig.getParameter<std::string>("fileName");
    m_shiftErr = iConfig.getParameter<double>("shiftErr");
    m_angleErr = iConfig.getParameter<double>("angleErr");
  }
  else
  {
    throw cms::Exception("BadConfig") << "input must be \"ideal\", \"db\", \"surveydb\", or \"xml\".\n";
  }

  if (m_output == std::string("none")) {}
  else if (m_output == std::string("db")) {}
  else if (m_output == std::string("surveydb")) {}
  else if (m_output == std::string("xml"))
  {
    m_outputXML = iConfig.getParameter<edm::ParameterSet>("outputXML");
  }
  else
  {
    throw cms::Exception("BadConfig") << "output must be \"none\", \"db\", or \"surveydb\".\n";
  }
}


void MuonGeometryDBConverter::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup)
{
  if (!m_done)
  {
    MuonAlignment *muonAlignment = NULL;

    if (m_input == std::string("ideal"))
    {
      MuonAlignmentInputMethod inputMethod;
      muonAlignment = new MuonAlignment(iSetup, inputMethod);
      muonAlignment->fillGapsInSurvey(0., 0.);
    }
    else if (m_input == std::string("db"))
    {
      MuonAlignmentInputDB inputMethod(m_dtLabel, m_cscLabel, m_getAPEs);
      muonAlignment = new MuonAlignment(iSetup, inputMethod);
      if (m_getAPEs)  muonAlignment->copyAlignmentToSurvey(m_shiftErr, m_angleErr);
    }
    else if (m_input == std::string("surveydb"))
    {
      MuonAlignmentInputSurveyDB inputMethod(m_dtLabel, m_cscLabel);
      muonAlignment = new MuonAlignment(iSetup, inputMethod);
      muonAlignment->copySurveyToAlignment();
    }
    else if (m_input == std::string("scenario"))
    {
      MuonAlignmentInputMethod inputMethod;
      muonAlignment = new MuonAlignment(iSetup, inputMethod);

      MuonScenarioBuilder muonScenarioBuilder(muonAlignment->getAlignableMuon());
      muonScenarioBuilder.applyScenario(m_misalignmentScenario);
      muonAlignment->copyAlignmentToSurvey(m_shiftErr, m_angleErr);
    }
    else if (m_input == std::string("xml"))
    {
      MuonAlignmentInputXML inputMethod(m_fileName);
      muonAlignment = new MuonAlignment(iSetup, inputMethod);
      muonAlignment->fillGapsInSurvey(m_shiftErr, m_angleErr);
    }

    /////////////

    if (muonAlignment)
    {
      if (m_output == std::string("none")) {}
      else if (m_output == std::string("db"))       muonAlignment->saveToDB();
      else if (m_output == std::string("surveydb")) muonAlignment->saveSurveyToDB();
      else if (m_output == std::string("xml"))      muonAlignment->writeXML(m_outputXML, iSetup);
      delete muonAlignment;
    }

    m_done = true;
  } // end if not done
  else  throw cms::Exception("BadConfig") << "Set maxEvents.input to 1.  (Your output is okay.)\n";
}

//define this as a plug-in
DEFINE_FWK_MODULE(MuonGeometryDBConverter);
