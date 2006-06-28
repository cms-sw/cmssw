#include "DQM/DTMonitorClient/interface/DTWebInterface.h"

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
DTWebInterface::DTWebInterface(std::string theContextURL, std::string theApplicationURL, MonitorUserInterface ** _mui_p)
  : WebInterface(theContextURL, theApplicationURL, _mui_p) {
  
  performCheckingQTGlobalStatus=false;
  performCheckingQTDetailedStatus=false;
  performCheckingNoiseStatus=false;

  Navigator * nav = new Navigator(getApplicationURL(), "50px", "10px");
  ContentViewer * cont = new ContentViewer(getApplicationURL(), "180px", "10px");

  Button * butQTGlobal = new Button(getApplicationURL(), 
				    "310px", "10px", "QTGlobalStatus", "Check QT Global Status");
  Button * butQTGlobalStop = new Button(getApplicationURL(), 
					"340px", "10px", "QTGlobalStatusStop", "Stop Checking QT Global Status");

  Button * butQTDetailed = new Button(getApplicationURL(), 
				      "390px", "10px", "QTDetailedStatus ", "Check QT Detailed Status");
  Button * butQTDetailedStop = new Button(getApplicationURL(), 
					  "420px", "10px", "QTDetailedStatusStop", "Stop Checking QT Detailed Status");


  Button * butNoiseCheck = new Button(getApplicationURL(), 
				      "310px", "10px", "NoiseCheck", "Check Noise");
  Button * butNoiseCheckStop = new Button(getApplicationURL(), 
					  "340px", "10px", "NoiseCheckStop", "Stop Noise Checking");


  GifDisplay * dis = new GifDisplay(getApplicationURL(), 
				    "50px","630px", "500px", "800px", "MyGifDisplay");
  GifDisplay * dis2 = new GifDisplay(getApplicationURL(), 
				     "650px", "10px", "500px", "800px", "MyOtherGifDisplay");
  
  page_p = new WebPage(getApplicationURL());
 
  page_p->add("navigator", nav);
  page_p->add("contentViewer", cont);
  page_p->add("butQTGlobalCheck", butQTGlobal);
  page_p->add("butQTDetailedCheck", butQTDetailed);
  page_p->add("QTGlobalStatusStop", butQTGlobalStop);
  page_p->add("QTDetailedStatusStop", butQTDetailedStop);

  page_p->add("NoiseCheck", butNoiseCheck);
  page_p->add("NoiseCheckStop", butNoiseCheckStop);

  page_p->add("gifDisplay", dis);
  page_p->add("otherGifDisplay", dis2);
}

/*
  Only implement the handleCustomRequest function if you have widgets that invoke 
  custom-made methods defined in your client. In this example we have created a 
  Button that makes custom requests, therefore we need to implement it.
*/
void DTWebInterface::handleCustomRequest(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception)
{
  // this is the way to get the string that identifies the request:
  std::multimap<std::string, std::string> request_multimap;
  CgiReader reader(in);
  reader.read_form(request_multimap);
  std::string requestID = get_from_multimap(request_multimap, "RequestID");

  // if you have more than one custom requests, add 'if' statements accordingly:
  if (requestID == "QTGlobalStatus")       checkQTGlobalStatus(in, out,true);
  if (requestID == "QTDetailedStatus")     checkQTDetailedStatus(in, out,true);
  if (requestID == "QTGlobalStatusStop")   checkQTGlobalStatus(in, out,false);
  if (requestID == "QTDetailedStatusStop") checkQTDetailedStatus(in, out,false);

  if (requestID == "NoiseCheck")       checkNoiseStatus(in, out,true);
  if (requestID == "NoiseCheckStop")   checkNoiseStatus(in, out,false);

}


void DTWebInterface::checkQTGlobalStatus(xgi::Input * in, xgi::Output * out, bool start ) 
  throw (xgi::exception::Exception)
{
  performCheckingQTGlobalStatus=start;
}

void DTWebInterface::checkQTDetailedStatus(xgi::Input * in, xgi::Output * out, bool start ) 
  throw (xgi::exception::Exception)
{
  performCheckingQTDetailedStatus=start;
}


void DTWebInterface::checkNoiseStatus(xgi::Input * in, xgi::Output * out, bool start ) 
  throw (xgi::exception::Exception)
{
  performCheckingNoiseStatus=start;
}
