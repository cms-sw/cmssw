#include "DQMServices/Examples/interface/ExampleWebInterface.h"

#include "DQMServices/WebComponents/interface/Button.h"
#include "DQMServices/WebComponents/interface/CgiWriter.h"
#include "DQMServices/WebComponents/interface/CgiReader.h"
#include "DQMServices/WebComponents/interface/ConfigBox.h"
#include "DQMServices/WebComponents/interface/Navigator.h"
#include "DQMServices/WebComponents/interface/ContentViewer.h"
#include "DQMServices/WebComponents/interface/GifDisplay.h"

ExampleWebInterface::ExampleWebInterface(std::string theContextURL, std::string theApplicationURL, MonitorUserInterface ** _mui_p)
  : WebInterface(theContextURL, theApplicationURL, _mui_p)
{
  Navigator * nav = new Navigator(getApplicationURL(), "50px", "10px");
  ContentViewer * cont = new ContentViewer(getApplicationURL(), "180px", "10px");
  Button * but = new Button(getApplicationURL(), "310px", "10px", "MyCustomRequest", "Submit a custom request");
  GifDisplay * dis = new GifDisplay(getApplicationURL(), "50px","350px", "200px", "300px", "MyGifDisplay");
  GifDisplay * dis2 = new GifDisplay(getApplicationURL(), "50px", "700px", "200px", "300px", "MyOtherGifDisplay");
  
  page_p = new WebPage(getApplicationURL());
  page_p->add("navigator", nav);
  page_p->add("contentViewer", cont);
  page_p->add("button", but);
  page_p->add("gifDisplay", dis);
  page_p->add("otherGifDisplay", dis2);
}

void ExampleWebInterface::handleCustomRequest(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception)
{
  std::multimap<std::string, std::string> request_multimap;
  CgiReader reader(in);
  reader.read_form(request_multimap);

  // get the string that identifies the request:
  std::string requestID = get_from_multimap(request_multimap, "RequestID");

  if (requestID == "MyCustomRequest") CustomRequestResponse(in, out);
}

void ExampleWebInterface::CustomRequestResponse(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception)
{
  std::cout << "A custom request has arrived" << std::endl;
}
