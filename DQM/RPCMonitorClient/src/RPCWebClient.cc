/** \file
 *
 *  Implementation of RPCWebClient
 *
 *  $Date: 2006/01/25 16:29:11 $
 *  $Revision: 1.4 $
 *  \author Ilaria Segoni
 */
#include "DQM/RPCMonitorClient/interface/RPCWebClient.h"
#include "DQMServices/WebComponents/interface/Button.h"
#include "DQMServices/WebComponents/interface/CgiWriter.h"
#include "DQMServices/WebComponents/interface/CgiReader.h"
#include "DQMServices/WebComponents/interface/ConfigBox.h"
#include "DQMServices/WebComponents/interface/Navigator.h"
#include "DQMServices/WebComponents/interface/ContentViewer.h"
#include "DQMServices/WebComponents/interface/GifDisplay.h"

#include "DQMServices/QualityTests/interface/QCriterionRoot.h"

RPCWebClient::RPCWebClient(xdaq::ApplicationStub * s) : DQMWebClient(s)
{
  printout=true;
  testsWereSet=false;
  qualityTests=new RPCQualityTester();
  
  ConfigBox * box = new ConfigBox(getApplicationURL(), "50px", "50px");
  Navigator * nav = new Navigator(getApplicationURL(), "1000px", "50px");
  ContentViewer * cont = new ContentViewer(getApplicationURL(), "1200px", "50px");
  
///Display for basic plots:
  GifDisplay * disBasePlots = new GifDisplay(getApplicationURL(), "50px","400px", "600px", "800px", "BasePlotsGifDisplay");   


///Display for on-line requests
  GifDisplay * dis = new GifDisplay(getApplicationURL(), "1000px","700px", "400px", "600px", "MyGifDisplay"); 
 
///Display for reference/blessed plots:
  GifDisplay * disRefPlots = new GifDisplay(getApplicationURL(), "1500px","50px", "600px", "800px", "RefPlotsGifDisplay");   

  Button * butSetQT = new Button(getApplicationURL(), "480px", "50px", "SetUpQltyTests", "Set Up Quality Tests");
  Button * butCheckQT = new Button(getApplicationURL(), "510px", "50px", "CheckQltyTests", "Check Quality Tests");

  page = new WebPage(getApplicationURL());
  page->add("configBox", box);
  page->add("navigator", nav);
  page->add("contentViewer", cont);
  page->add("button_setTests", butSetQT);
  page->add("button_checkTests", butCheckQT);
  page->add("gifDisplay", dis);
  page->add("gifDisplayBase", disBasePlots);
  page->add("gifDisplayRef", disRefPlots);

}


void RPCWebClient::Default(xgi::Input * in, xgi::Output * out ) 
  throw (xgi::exception::Exception)
{
  string pagetitle = "RPCWebClient";
  CgiWriter writer(out, getContextURL());
  writer.output_preamble();
  writer.output_head();
  page->printHTML(out);
  writer.output_finish();
}

 

void RPCWebClient::Request(xgi::Input * in, xgi::Output * out ) 
  throw (xgi::exception::Exception)
{
  // put the request information in a multimap...
  std::multimap<string, string> request_multimap;
  CgiReader reader(in);
  reader.read_form(request_multimap);

  // get the string that identifies the request:
  std::string requestID = this->get_from_multimap(request_multimap, "RequestID");

  if (requestID == "SetUpQltyTests")  this->SetupQTestsRequest(in, out);
  if (requestID == "CheckQltyTests")  this->CheckQTestRequest(in, out);
}


void RPCWebClient::SetupQTestsRequest(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception)
{

  if(testsWereSet){   
    if(printout) std::cout<< "Quality tests are already set!"<<std::endl;
    return;     
  } 
  
 if(printout) std::cout << "Quality Tests are being set up" << std::endl;
 
  qualityTests->SetupTests(mui);
 //qualityTests->SetupTests(mui);
 //qualityTests->AttachTests(mui);
 
 testsWereSet=true;
 return;

}




void RPCWebClient::CheckQTestRequest(xgi::Input * in, xgi::Output *out) throw (xgi::exception::Exception)
{
  if(!testsWereSet){   
    if(printout) cout<< "No quality tests are set. Set Quality tests First!"<<endl;
    return;     
  }

  qualityTests->CheckTests(mui);

  return;
}


//
// provides factory method for instantion of SimpleWeb application
//
XDAQ_INSTANTIATOR_IMPL(RPCWebClient)
  
