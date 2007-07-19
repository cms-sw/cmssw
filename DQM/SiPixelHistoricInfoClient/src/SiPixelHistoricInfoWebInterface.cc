#include "DQMServices/WebComponents/interface/Button.h"
#include "DQMServices/WebComponents/interface/CgiWriter.h"
#include "DQMServices/WebComponents/interface/CgiReader.h"
#include "DQMServices/WebComponents/interface/ConfigBox.h"
#include "DQMServices/WebComponents/interface/Navigator.h"
#include "DQMServices/WebComponents/interface/ContentViewer.h"
#include "DQMServices/WebComponents/interface/GifDisplay.h"

#include "DQM/SiPixelHistoricInfoClient/interface/SiPixelHistoricInfoWebInterface.h"


// to create your widgets in the constructor of your web interface
SiPixelHistoricInfoWebInterface::SiPixelHistoricInfoWebInterface(std::string theContextURL, 
                                                                 std::string theApplicationURL, 
								 MonitorUserInterface** _mui_p)
                               : WebInterface(theContextURL, theApplicationURL, _mui_p) {
  // a navigator allows you to make subscriptions:
  // Navigator* nav = new Navigator(getApplicationURL(), "20px", "10px");
  // a content viewer allows you to select ME's to draw:
  ContentViewer* cont = new ContentViewer(getApplicationURL(), "20px", "10px");
  
  Button* saveBut = new Button(getApplicationURL(), "300px", "10px", "SaveToFile", "Save To File");

  // two inline frames to display plots:
  GifDisplay* dis = new GifDisplay(getApplicationURL(), "20px","200px", "500px", "700px", "MyGifDisplay");
  // every web interface needs to instantiate a webpage...
  page_p = new WebPage(getApplicationURL());

  // and add its widgets to it:
  // page_p->add("navigator", nav); // maybe this client does not need a navigator
  page_p->add("contentViewer", cont);
  page_p->add("SvButton", saveBut);
  page_p->add("gifDisplay", dis);
}


// only implement the handleCustomRequest function if you have widgets that invoke custom-made 
// methods defined in your client. In this example we have created a Button that makes custom 
// requests, therefore we need to implement it.
void SiPixelHistoricInfoWebInterface::handleCustomRequest(xgi::Input * in, xgi::Output * out) 
                                      throw (xgi::exception::Exception) {
  // this is the way to get the string that identifies the request:
  std::multimap<std::string, std::string> request_multimap;
  CgiReader reader(in);
  reader.read_form(request_multimap);
  std::string requestID = get_from_multimap(request_multimap, "RequestID");

  // if you have more than one custom requests, add 'if' statements accordingly:
  if (requestID=="SaveToFile") saveToFile(in, out);
}


void SiPixelHistoricInfoWebInterface::saveToFile(xgi::Input * in, xgi::Output *out) 
                                      throw (xgi::exception::Exception) {
  if (!getSaveToFile()) { // set to true if not already true
    std::cout << "SiPixelHistoricInfoWebInterface::saveToFile: put request for saving Monitoring Elements in file." << std::endl;
    setSaveToFile(true);
  }
  return;
}
