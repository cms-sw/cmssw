/*
 * \file EcalEndcapMonitorXdaqClient.h
 cc
 * $Date: 2009/02/27 13:54:04 $
 * $Revision: 1.114 $
 * \author G. Della Ricca
 *
*/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "OnlineDB/EcalCondDB/interface/RunDat.h"
#include "OnlineDB/EcalCondDB/interface/MonRunDat.h"

#include "DQM/EcalEndcapMonitorClient/interface/EcalEndcapMonitorClient.h"

#include "EventFilter/Utilities/interface/ModuleWeb.h"

#include "xgi/Input.h"
#include "xgi/Output.h"

#include "xgi/Method.h"
#include "xgi/Utils.h"

#include "cgicc/Cgicc.h"
#include "cgicc/FormEntry.h"
#include "cgicc/HTMLClasses.h"

class EcalEndcapMonitorXdaqClient: public EcalEndcapMonitorClient, public evf::ModuleWeb{

public:

/// Constructor
EcalEndcapMonitorXdaqClient(const edm::ParameterSet & ps) : EcalEndcapMonitorClient(ps), ModuleWeb("EcalEndcapMonitorXdaqClient") {};

/// Destructor
virtual ~EcalEndcapMonitorXdaqClient() {};

/// XDAQ web page
void defaultWebPage(xgi::Input *in, xgi::Output *out) {

  string path;
  string mname;

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

  } catch (exception &e) {

    cerr << "Standard C++ exception : " << e.what() << endl;

  }

  *out << cgicc::HTMLDoctype(cgicc::HTMLDoctype::eStrict)            << endl;
  *out << cgicc::html().set("lang", "en").set("dir","ltr")           << endl;

  *out << "<html>"                                                   << endl;

  *out << "<head>"                                                   << endl;

  *out << "<title>" << typeid(EcalEndcapMonitorXdaqClient).name()
       << " MAIN</title>"                                            << endl;

  if ( autorefresh_ ) {
    *out << "<meta http-equiv=\"refresh\" content=\"3\">"            << endl;
  }

  *out << "</head>"                                                  << endl;

  *out << "<body>"                                                   << endl;

  *out << cgicc::form().set("method","GET").set("action", path )
       << endl;
  *out << cgicc::input().set("type","hidden").set("name","module").set("value", mname)
       << endl;
  *out << cgicc::input().set("type","hidden").set("name","autorefresh").set("value", autorefresh_?"0":"1")
       << endl;
  *out << cgicc::input().set("type","submit").set("value",autorefresh_?"Toggle AutoRefresh OFF":"Toggle AutoRefresh ON")
       << endl;
  *out << cgicc::form()                                              << endl;

  *out << cgicc::h3( "EcalEndcapMonitorXdaqClient Status" ).set( "style", "font-family:arial" ) << endl;

  *out << "<table style=\"font-family: arial\"><tr><td>" << endl;

  *out << "<p style=\"font-family: arial\">"
       << "<table border=1>"
       << "<tr><th>Cycle</th><td align=right>" << ievt_;
  int nevt = 0;
  if ( h_ != 0 ) nevt = int( h_->GetEntries());
  *out << "<tr><th>Event</th><td align=right>" << nevt
       << "</td><tr><th>Run</th><td align=right>" << run_
       << "</td><tr><th>Run Type</th><td align=right> " << this->getRunType()
       << "</td></table></p>" << endl;

  *out << "</td><td>" << endl;

  *out << "<p style=\"font-family: arial\">"
       << "<table border=1>"
       << "<tr><th>Evt Type</th><th>Evt/Run</th><th>Evt Type</th><th>Evt/Run</th>" << endl;
  for( unsigned int i = 0, j = 0; i < runTypes_.size(); i++ ) {
    if ( strcmp(runTypes_[i].c_str(), "UNKNOWN") != 0 ) {
      if ( j++%2 == 0 ) *out << "<tr>";
      nevt = 0;
      if ( h_ != 0 ) nevt = int( h_->GetBinContent(i+1) );
      *out << "<td>" << runTypes_[i]
           << "</td><td align=right>" << nevt << endl;
    }
  }
  *out << "</td></table></p>" << endl;

  *out << "</td><tr><td colspan=2>" << endl;

  *out << "<p style=\"font-family: arial\">"
       << "<table border=1>"
       << "<tr><th>Client</th><th>Cyc/Job</th><th>Cyc/Run</th><th>Client</th><th>Cyc/Job</th><th>Cyc/Run</th>" << endl;
  for( unsigned int i = 0; i < clients_.size(); i++ ) {
    if ( clients_[i] != 0 ) {
      if ( i%2 == 0 ) *out << "<tr>";
      *out << "<td>" << clientsNames_[i]
           << "</td><td align=right>" << clients_[i]->getEvtPerJob()
           << "</td><td align=right>" << clients_[i]->getEvtPerRun() << endl;
    }
  }
  *out << "</td></table></p>" << endl;

  *out << "</td><tr><td>" << endl;


  *out << "<p style=\"font-family: arial\">"
       << "<table border=1>"
       << "<tr><th colspan=2>RunIOV</th>"
       << "<tr><td>Run Number</td><td align=right> " << runiov_.getRunNumber()
       << "</td><tr><td>Run Start</td><td align=right> " << runiov_.getRunStart().str()
       << "</td><tr><td>Run End</td><td align=right> " << runiov_.getRunEnd().str()
       << "</td></table></p>" << endl;

  *out << "</td><td colsapn=2>" << endl;

  *out << "<p style=\"font-family: arial\">"
       << "<table border=1>"
       << "<tr><th colspan=2>RunTag</th>"
       << "<tr><td>GeneralTag</td><td align=right> " << runiov_.getRunTag().getGeneralTag()
       << "</td><tr><td>Location</td><td align=right> " << runiov_.getRunTag().getLocationDef().getLocation()
       << "</td><tr><td>Run Type</td><td align=right> " << runiov_.getRunTag().getRunTypeDef().getRunType()
       << "</td></table></p>" << endl;

  *out << "</td><tr><td>" << endl;

  *out << "<p style=\"font-family: arial\">"
       << "<table border=1>"
       << "<tr><th colspan=2>MonRunIOV</th>"
       << "<tr><td>SubRun Number</td><td align=right> " << moniov_.getSubRunNumber()
       << "</td><tr><td>SubRun Start</td><td align=right> " << moniov_.getSubRunStart().str()
       << "</td><tr><td>SubRun End</td><td align=right> " << moniov_.getSubRunEnd().str()
       << "</td></table></p>" << endl;

  *out << "</td><td colspan=2>" << endl;

  *out << "<p style=\"font-family: arial\">"
       << "<table border=1>"
       << "<tr><th colspan=2>MonRunTag</th>"
       << "<tr><td>GeneralTag</td><td align=right> " << moniov_.getMonRunTag().getGeneralTag()
       << "</td><tr><td>Monitoring Version</td><td align=right> " << moniov_.getMonRunTag().getMonVersionDef().getMonitoringVersion()
       << "</td></table></p>" << endl;

  *out << "</td><table>" << endl;


  *out << "</body>"                                                  << endl;

  *out << "</html>"                                                  << endl;

  };

void publish(xdata::InfoSpace *) {};

};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EcalEndcapMonitorXdaqClient);

