/*
 * \file SiStripAnalyser.cc
 * 
 * $Date: 2007/07/16 20:19:21 $
 * $Revision: 1.3 $
 * \author  S. Dutta INFN-Pisa
 *
 */


#include "DQM/SiStripMonitorClient/interface/SiStripAnalyser.h"

// Framework
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/UI/interface/MonitorUIRoot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/WebComponents/interface/Button.h"
#include "DQMServices/WebComponents/interface/CgiWriter.h"
#include "DQMServices/WebComponents/interface/CgiReader.h"
#include "DQMServices/WebComponents/interface/ConfigBox.h"
#include "DQMServices/WebComponents/interface/WebPage.h"

#include "CondFormats/DataRecord/interface/SiStripFedCablingRcd.h"
#include "CondFormats/SiStripObjects/interface/SiStripFedCabling.h"

#include "DQM/SiStripMonitorClient/interface/SiStripWebInterface.h"
#include "DQM/SiStripMonitorClient/interface/TrackerMapCreator.h"
#include "DQM/SiStripMonitorClient/interface/SiStripUtility.h"

#include <SealBase/Callback.h>

#include "xgi/Method.h"
#include "xgi/Utils.h"

#include "cgicc/Cgicc.h"
#include "cgicc/FormEntry.h"
#include "cgicc/HTMLClasses.h"

#include <iostream>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;
//
// -- Constructor
//
SiStripAnalyser::SiStripAnalyser(const edm::ParameterSet& ps) :
  ModuleWeb("SiStripAnalyser"){
  
  edm::LogInfo("SiStripAnalyser") <<  " Creating SiStripAnalyser " << "\n" ;
  parameters = ps;
  
  dbe = edm::Service<DaqMonitorBEInterface>().operator->();

  tkMapFrequency_   = -1;
  summaryFrequency_ = -1;
  fileSaveFrequency_ = parameters.getUntrackedParameter<int>("FileSaveFrequency",50); 

  // instantiate Monitor UI without connecting to any monitoring server
  // (i.e. "standalone mode")
  mui_ = new MonitorUIRoot();

  // instantiate web interface
  sistripWebInterface_ = new SiStripWebInterface("dummy", "dummy", &mui_);
  //  xgi::bind(this, &SiStripAnalyser::handleWebRequest, "Request");
  
}
//
// -- Destructor
//
SiStripAnalyser::~SiStripAnalyser(){

  edm::LogInfo("SiStripAnalyser") <<  " Deleting SiStripAnalyser " << "\n" ;
  if (sistripWebInterface_) delete sistripWebInterface_;
  if (trackerMapCreator_) delete trackerMapCreator_;

}
//
// -- End Job
//
void SiStripAnalyser::endJob(){

  cout << " Saving Monitoring Elements " << endl;
  saveAll();
}
//
// -- Begin Job
//
void SiStripAnalyser::beginJob(const edm::EventSetup& eSetup){

  nevents = 0;
  runNumber_ = 0;
  sistripWebInterface_->readConfiguration(summaryFrequency_);
  edm::LogInfo("SiStripAnalyser") << " Configuration files read out correctly" 
                                  << "\n" ;
  cout  << " Update Frequencies are " << tkMapFrequency_ << " " 
                                      << summaryFrequency_ << endl ;

  collationFlag_ = parameters.getUntrackedParameter<int>("CollationtionFlag",0); 

  // Get Fed cabling
  eSetup.get<SiStripFedCablingRcd>().get(fedCabling_);
  trackerMapCreator_ = new TrackerMapCreator();
  if (trackerMapCreator_->readConfiguration()) {
    tkMapFrequency_ = trackerMapCreator_->getFrequency();
  }
}
//
//  -- Analyze 
//
void SiStripAnalyser::analyze(const edm::Event& e, const edm::EventSetup& eSetup){
  nevents++;
  runNumber_ = e.id().run();

  eSetup.get<SiStripFedCablingRcd>().get(fedCabling_);
 
  if (nevents <= 3) return;

  cout << " ===> Iteration #" << nevents << endl;
  // -- Create summary monitor elements according to the frequency
  if (summaryFrequency_ != -1 && nevents%summaryFrequency_ == 1) {
    cout << " Creating Summary " << endl;
    sistripWebInterface_->setActionFlag(SiStripWebInterface::Summary);
    sistripWebInterface_->performAction();
  }
  // -- Create TrackerMap  according to the frequency
  if (tkMapFrequency_ != -1 && nevents%tkMapFrequency_ == 1) {
    cout << " Creating Tracker Map " << endl;
    //    trackerMapCreator_->create(mui_);
    trackerMapCreator_->create(fedCabling_, mui_);
  }
  // Create predefined plots
  if (nevents%10  == 1) {
    cout << " Creating predefined plots " << endl;
    sistripWebInterface_->setActionFlag(SiStripWebInterface::PlotHistogramFromLayout);
    sistripWebInterface_->performAction();
  }

  if (nevents % fileSaveFrequency_ == 1) {
    saveAll();
  }
}
//
// -- Save file
//
void SiStripAnalyser::saveAll() {
  ostringstream fname;
  fname << "SiStripWebInterface_" << runNumber_ << ".root";
  cout << " Saving Monitoring Elements in " << fname.str() <<endl;
  sistripWebInterface_->setOutputFileName(fname.str());
  sistripWebInterface_->setActionFlag(SiStripWebInterface::SaveData);
  sistripWebInterface_->performAction();
}
//
// -- Create default web page
//
void SiStripAnalyser::defaultWebPage(xgi::Input *in, xgi::Output *out)
{
      std::string path;
      std::string mname;
      try 
	{
           cgicc::Cgicc cgi(in);
	  //	  if ( xgi::Utils::hasFormElement(cgi,"mybut") )
	  cgicc::CgiEnvironment cgie(in);
	  path = cgie.getPathInfo() + "?" + cgie.getQueryString();
	}
      catch (const std::exception & e) 
    {
      // don't care if it did not work
    }

      /*
  *out << "<html>"                                                   << endl;
  *out << "<head>"                                                   << endl;

  *out << "<title>" << typeid(SiStripAnalyser).name()
       << " MAIN</title>"                                             << endl;
  *out << "<script type=\"text/javascript\" language=\"JavaScript\">" << endl;
  *out << "alert(window.location.href);"                              << endl;
  *out << "window.location=\"http://lxplus208.cern.ch:40001/temporary/Online.html\";"                << endl; 
  *out << "</script>"                                                 << endl;
  *out << "</head>"                                                  << endl;
  *out << "</html>"                                                  << endl;
  */

    static const int BUF_SIZE = 256;
    ifstream fin("loader.html", ios::in);
    if (!fin) {
      cerr << "Input File: loader.html"<< " could not be opened!" << endl;
      return;
    }
    char buf[BUF_SIZE];
    ostringstream html_dump;
    while (fin.getline(buf, BUF_SIZE, '\n')) { // pops off the newline character 
     html_dump << buf << std::endl;
    }
    fin.close();

    *out << html_dump.str() << std::endl;

   
      /*  using std::endl;
  *out << "<html>"                                                   << endl;
  *out << "<head>"                                                   << endl;

  *out << "<title>" << typeid(SiStripAnalyser).name()
       << " MAIN</title>"                                            << endl;
  *out << "</head>"                                                  << endl;
  *out << "<body>"                                                   << endl;
  *out << "SiStripClient" << " Default web page "                    << endl;
  *out << "</body>"                                                  << endl;
  *out << "</html>"                                                  << endl;

  *out << cgicc::form().set("method","GET").set("action", "javascript:void%200") 
       << std::endl;
  *out << cgicc::input().set("type","submit").set("value","test").set("onclick", "javascript:alert('hello');return false;")
       << std::endl;
  *out << cgicc::form()						   
       << std::endl;  
  *out << "<p>Run: " << runNumber_ << " Total updates: " << nevents << std::endl;

  *out << "<\br> "<< std::endl;
  *out << cgicc::a().set("href", "temporary/Online.html") << "MyPage" << cgicc::a() << std::endl;


  *out << "</body>"
       << std::endl;
  *out << "</html>" 
  << std::endl;*/

}
//
// Handles all HTTP requests of the form ..../Request?RequestID=
//
//void SiStripAnalyser::handleWebRequest(xgi::Input *in, xgi::Output *out) {
//  sistripWebInterface_->handleCustomRequest(in, out);
//}
