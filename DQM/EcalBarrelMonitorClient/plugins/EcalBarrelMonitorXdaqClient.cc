/*
 * \file EcalBarrelMonitorXdaqClient.cc
 *
 * $Date: 2010/03/27 20:30:36 $
 * $Revision: 1.4 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#ifdef WITH_ECAL_COND_DB
#include "OnlineDB/EcalCondDB/interface/RunDat.h"
#include "OnlineDB/EcalCondDB/interface/MonRunDat.h"
#endif

#include "DQM/EcalBarrelMonitorClient/interface/EcalBarrelMonitorClient.h"

#include "EventFilter/Utilities/interface/ModuleWeb.h"

#include "xgi/Input.h"
#include "xgi/Output.h"

#include "xgi/Method.h"
#include "xgi/Utils.h"

#include "cgicc/Cgicc.h"
#include "cgicc/FormEntry.h"
#include "cgicc/HTMLClasses.h"

class EcalBarrelMonitorXdaqClient: public EcalBarrelMonitorClient, public evf::ModuleWeb{

public:

/// Constructor
EcalBarrelMonitorXdaqClient(const edm::ParameterSet & ps) : EcalBarrelMonitorClient(ps), ModuleWeb("EcalBarrelMonitorXdaqClient") {};

/// Destructor
virtual ~EcalBarrelMonitorXdaqClient() {};

/// XDAQ web page
void defaultWebPage(xgi::Input *in, xgi::Output *out) {

  std::string path;
  std::string mname;

  static bool autorefresh_ = false;

  try {

    cgicc::Cgicc cgi(in);

    if ( xgi::Utils::hasFormElement(cgi,"autorefresh") ) {
      autorefresh_ = xgi::Utils::getFormElement(cgi, "autorefresh")->getIntegerValue() != 0;
    }

    if ( xgi::Utils::hasFormElement(cgi,"module") ) {
      mname = xgi::Utils::getFormElement(cgi, "module")->getValue();
    }

    cgicc::CgiEnvironment cgie(in);
    path = cgie.getPathInfo() + "?" + cgie.getQueryString();

  } catch (std::exception &e) {

    std::cerr << "Standard C++ exception : " << e.what() << std::endl;

  }

  *out << cgicc::HTMLDoctype(cgicc::HTMLDoctype::eStrict)            << std::endl;
  *out << cgicc::html().set("lang", "en").set("dir","ltr")           << std::endl;

  *out << "<html>"                                                   << std::endl;

  *out << "<head>"                                                   << std::endl;

  *out << "<title>" << typeid(EcalBarrelMonitorXdaqClient).name()
       << " MAIN</title>"                                            << std::endl;

  if ( autorefresh_ ) {
    *out << "<meta http-equiv=\"refresh\" content=\"3\">"            << std::endl;
  }

  *out << "</head>"                                                  << std::endl;

  *out << "<body>"                                                   << std::endl;

  *out << cgicc::form().set("method","GET").set("action", path )
       << std::endl;
  *out << cgicc::input().set("type","hidden").set("name","module").set("value", mname)
       << std::endl;
  *out << cgicc::input().set("type","hidden").set("name","autorefresh").set("value", autorefresh_?"0":"1")
       << std::endl;
  *out << cgicc::input().set("type","submit").set("value",autorefresh_?"Toggle AutoRefresh OFF":"Toggle AutoRefresh ON")
       << std::endl;
  *out << cgicc::form()                                              << std::endl;

  *out << cgicc::h3( "EcalBarrelMonitorXdaqClient Status" ).set( "style", "font-family:arial" ) << std::endl;

  *out << "<table style=\"font-family: arial\"><tr><td>" << std::endl;

  *out << "<p style=\"font-family: arial\">"
       << "<table border=1>"
       << "<tr><th>Cycle</th><td align=right>" << ievt_;
  int nevt = 0;
  if ( h_ != 0 ) nevt = int( h_->GetEntries());
  *out << "<tr><th>Event</th><td align=right>" << nevt
       << "</td><tr><th>Run</th><td align=right>" << run_
       << "</td><tr><th>Run Type</th><td align=right> " << this->getRunType()
       << "</td></table></p>" << std::endl;

  *out << "</td><td>" << std::endl;

  *out << "<p style=\"font-family: arial\">"
       << "<table border=1>"
       << "<tr><th>Evt Type</th><th>Evt/Run</th><th>Evt Type</th><th>Evt/Run</th>" << std::endl;
  for( unsigned int i = 0, j = 0; i < runTypes_.size(); i++ ) {
    if ( strcmp(runTypes_[i].c_str(), "UNKNOWN") != 0 ) {
      if ( j++%2 == 0 ) *out << "<tr>";
      nevt = 0;
      if ( h_ != 0 ) nevt = int( h_->GetBinContent(i+1) );
      *out << "<td>" << runTypes_[i]
           << "</td><td align=right>" << nevt << std::endl;
    }
  }
  *out << "</td></table></p>" << std::endl;

  *out << "</td><tr><td colspan=2>" << std::endl;

  *out << "<p style=\"font-family: arial\">"
       << "<table border=1>"
       << "<tr><th>Client</th><th>Cyc/Job</th><th>Cyc/Run</th><th>Client</th><th>Cyc/Job</th><th>Cyc/Run</th>" << std::endl;
  for( unsigned int i = 0; i < clients_.size(); i++ ) {
    if ( clients_[i] != 0 ) {
      if ( i%2 == 0 ) *out << "<tr>";
      *out << "<td>" << clientsNames_[i]
           << "</td><td align=right>" << clients_[i]->getEvtPerJob()
           << "</td><td align=right>" << clients_[i]->getEvtPerRun() << std::endl;
    }
  }
  *out << "</td></table></p>" << std::endl;

  *out << "</td><tr><td>" << std::endl;

#ifdef WITH_ECAL_COND_DB
  *out << "<p style=\"font-family: arial\">"
       << "<table border=1>"
       << "<tr><th colspan=2>RunIOV</th>"
       << "<tr><td>Run Number</td><td align=right> " << runiov_.getRunNumber()
       << "</td><tr><td>Run Start</td><td align=right> " << runiov_.getRunStart().str()
       << "</td><tr><td>Run End</td><td align=right> " << runiov_.getRunEnd().str()
       << "</td></table></p>" << std::endl;

  *out << "</td><td colsapn=2>" << std::endl;

  *out << "<p style=\"font-family: arial\">"
       << "<table border=1>"
       << "<tr><th colspan=2>RunTag</th>"
       << "<tr><td>GeneralTag</td><td align=right> " << runiov_.getRunTag().getGeneralTag()
       << "</td><tr><td>Location</td><td align=right> " << runiov_.getRunTag().getLocationDef().getLocation()
       << "</td><tr><td>Run Type</td><td align=right> " << runiov_.getRunTag().getRunTypeDef().getRunType()
       << "</td></table></p>" << std::endl;

  *out << "</td><tr><td>" << std::endl;

  *out << "<p style=\"font-family: arial\">"
       << "<table border=1>"
       << "<tr><th colspan=2>MonRunIOV</th>"
       << "<tr><td>SubRun Number</td><td align=right> " << moniov_.getSubRunNumber()
       << "</td><tr><td>SubRun Start</td><td align=right> " << moniov_.getSubRunStart().str()
       << "</td><tr><td>SubRun End</td><td align=right> " << moniov_.getSubRunEnd().str()
       << "</td></table></p>" << std::endl;

  *out << "</td><td colspan=2>" << std::endl;

  *out << "<p style=\"font-family: arial\">"
       << "<table border=1>"
       << "<tr><th colspan=2>MonRunTag</th>"
       << "<tr><td>GeneralTag</td><td align=right> " << moniov_.getMonRunTag().getGeneralTag()
       << "</td><tr><td>Monitoring Version</td><td align=right> " << moniov_.getMonRunTag().getMonVersionDef().getMonitoringVersion()
       << "</td></table></p>" << std::endl;
#endif

  *out << "</td><table>" << std::endl;

  *out << "</body>"                                                  << std::endl;

  *out << "</html>"                                                  << std::endl;

  };

void publish(xdata::InfoSpace *) {};

};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EcalBarrelMonitorXdaqClient);

