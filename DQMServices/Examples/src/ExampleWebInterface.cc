#include "DQMServices/Examples/interface/ExampleWebInterface.h"

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
ExampleWebInterface::ExampleWebInterface(std::string theContextURL, std::string theApplicationURL, MonitorUserInterface ** _mui_p)
  : WebInterface(theContextURL, theApplicationURL, _mui_p)
{
  // a navigator allows you to make subscriptions:
  Navigator * nav = new Navigator(getApplicationURL(), "50px", "10px");

  // a content viewer allows you to select ME's to draw:
  ContentViewer * cont = new ContentViewer(getApplicationURL(), "180px", "10px");

  /* 
     This is a button that makes a request with RequestID = "MyCustomRequest".
     When you create such a button you need to define a function that gets called
     when such a request is made:
  */
  Button * but = new Button(getApplicationURL(), "310px", "10px", "MyCustomRequest", "Submit a custom request");

  // two inline frames that display plots:
  GifDisplay * dis = new GifDisplay(getApplicationURL(), "50px","350px", "200px", "300px", "MyGifDisplay");
  GifDisplay * dis2 = new GifDisplay(getApplicationURL(), "50px", "700px", "200px", "300px", "MyOtherGifDisplay");
  
  // every web interface needs to instantiate a WebPage...
  page_p = new WebPage(getApplicationURL());
  // ...and add its widgets to it:
  page_p->add("navigator", nav);
  page_p->add("contentViewer", cont);
  page_p->add("button", but);
  page_p->add("gifDisplay", dis);
  page_p->add("otherGifDisplay", dis2);
}

/*
  Only implement the handleCustomRequest function if you have widgets that invoke 
  custom-made methods defined in your client. In this example we have created a 
  Button that makes custom requests, therefore we need to implement it.
*/
void ExampleWebInterface::handleCustomRequest(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception)
{
  // this is the way to get the string that identifies the request:
  std::multimap<std::string, std::string> request_multimap;
  CgiReader reader(in);
  reader.read_form(request_multimap);
  std::string requestID = get_from_multimap(request_multimap, "RequestID");

  // if you have more than one custom requests, add 'if' statements accordingly:
  if (requestID == "MyCustomRequest") CustomRequestResponse(in, out);
}

/*
  Just a silly method that does nothing. Yours will be more meaningful.
*/
void ExampleWebInterface::CustomRequestResponse(xgi::Input * in, xgi::Output * out ) throw (xgi::exception::Exception)
{
  std::cout << "A custom request has arrived" << std::endl;
}
