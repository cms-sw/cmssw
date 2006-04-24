#include "DQM/RPCMonitorClient/interface/MuonWebInterface.h"

#include "DQMServices/WebComponents/interface/Button.h"
#include "DQMServices/WebComponents/interface/CgiWriter.h"
#include "DQMServices/WebComponents/interface/CgiReader.h"
#include "DQMServices/WebComponents/interface/ConfigBox.h"
#include "DQMServices/WebComponents/interface/Navigator.h"
#include "DQMServices/WebComponents/interface/ContentViewer.h"
#include "DQMServices/WebComponents/interface/GifDisplay.h"


/*
  Create your widgets in the constructor of your web interface
*/
MuonWebInterface::MuonWebInterface(std::string theContextURL, std::string theApplicationURL, MonitorUserInterface ** _mui_p)
  : WebInterface(theContextURL, theApplicationURL, _mui_p)
{
	checkQTGlobalStatus=false;
	checkQTDetailedStatus=false;
  	Navigator * nav = new Navigator(getApplicationURL(), "50px", "10px");
  	ContentViewer * cont = new ContentViewer(getApplicationURL(), "180px", "10px");

	Button * butQTGlobal = new Button(getApplicationURL(), "310px", "10px", "QTGlobalStatus", "Check Quality Tests Global Status");
  	Button * butQTDetailed = new Button(getApplicationURL(), "340px", "10px", "QTDetailedStatus ", "Check Quality Tests Detailed Status");
	Button * butQTGlobalStop = new Button(getApplicationURL(), "310px", "400px", "QTGlobalStatusStop", "Stop Checking QT Global Status");
  	Button * butQTDetailedStop = new Button(getApplicationURL(), "340px", "400px", "QTDetailedStatusStop", "Stop Checking QT Detailed Status");

	GifDisplay * dis = new GifDisplay(getApplicationURL(), "50px","350px", "200px", "300px", "MyGifDisplay");
	GifDisplay * dis2 = new GifDisplay(getApplicationURL(), "50px", "700px", "200px", "300px", "MyOtherGifDisplay");
  
	page_p = new WebPage(getApplicationURL());
 
        page_p->add("navigator", nav);
        page_p->add("contentViewer", cont);
        page_p->add("butQTGlobalCheck", butQTGlobal);
        page_p->add("butQTDetailedCheck", butQTDetailed);
        page_p->add("QTGlobalStatusStop", butQTGlobalStop);
        page_p->add("QTDetailedStatusStop", butQTDetailedStop);
        page_p->add("gifDisplay", dis);
        page_p->add("otherGifDisplay", dis2);
}

/*
  Only implement the handleCustomRequest function if you have widgets that invoke 
  custom-made methods defined in your client. In this example we have created a 
  Button that makes custom requests, therefore we need to implement it.
*/
void MuonWebInterface::handleCustomRequest(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception)
{
  // this is the way to get the string that identifies the request:
  std::multimap<std::string, std::string> request_multimap;
  CgiReader reader(in);
  reader.read_form(request_multimap);
  std::string requestID = get_from_multimap(request_multimap, "RequestID");

  // if you have more than one custom requests, add 'if' statements accordingly:
  if (requestID == "QTGlobalStatus")       this->CheckQTGlobalStatus(in, out,true);
  if (requestID == "QTDetailedStatus")     this->CheckQTDetailedStatus(in, out,true);
  if (requestID == "QTGlobalStatusStop")   this->CheckQTGlobalStatus(in, out,false);
  if (requestID == "QTDetailedStatusStop") this->CheckQTDetailedStatus(in, out,false);
}




void MuonWebInterface::CheckQTGlobalStatus(xgi::Input * in, xgi::Output * out, bool start ) throw (xgi::exception::Exception)
{
  std::cout << "Checking global status" << std::endl;
  checkQTGlobalStatus=start;
}

void MuonWebInterface::CheckQTDetailedStatus(xgi::Input * in, xgi::Output * out, bool start ) throw (xgi::exception::Exception)
{
  std::cout << "Checking Detailed Status" << std::endl;
  checkQTDetailedStatus=start;
  
}

