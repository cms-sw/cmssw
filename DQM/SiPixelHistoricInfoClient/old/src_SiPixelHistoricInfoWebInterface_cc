#include <iostream>
#include <map>

#include "DQMServices/WebComponents/interface/CgiReader.h"
#include "DQMServices/WebComponents/interface/ContentViewer.h"
#include "DQMServices/WebComponents/interface/Button.h"
#include "DQMServices/WebComponents/interface/GifDisplay.h"
#include "DQMServices/WebComponents/interface/HTMLLink.h"
#include "DQMServices/WebComponents/interface/WebPage.h"

#include "DQM/SiPixelHistoricInfoClient/interface/SiPixelHistoricInfoWebInterface.h"


SiPixelHistoricInfoWebInterface::SiPixelHistoricInfoWebInterface(std::string theContextURL, 
                                                                 std::string theApplicationURL, 
								 DQMOldReceiver** _mui_p)
                               : WebInterface(theContextURL, theApplicationURL, _mui_p) {
  ContentViewer* cv = new ContentViewer(getApplicationURL(),"20px","10px"); 
  Button* sb = new Button(getApplicationURL(),"300px","10px","SaveToFile","Save To File");
  Button* wb = new Button(getApplicationURL(),"300px","30px","WriteToDB","Write To DB");
  GifDisplay* gd = new GifDisplay(getApplicationURL(),"20px","200px","500px","700px","Gif Display");
  HTMLLink *hl = new HTMLLink(getApplicationURL(),"50px","50px","SiPixelHistoricInfoWebInterface",
                                                                "/temporary/Online.html");
  page_p = new WebPage(getApplicationURL());
  page_p->add("cntViewer", cv);
  page_p->add("svButton", sb);
  page_p->add("wrtButton", wb);
  page_p->add("gifDisplay", gd);
  page_p->add("htmlLink", hl);
}


SiPixelHistoricInfoWebInterface::~SiPixelHistoricInfoWebInterface() {}


void SiPixelHistoricInfoWebInterface::handleCustomRequest(xgi::Input* in, xgi::Output* out) 
                                      throw (xgi::exception::Exception) {
  CgiReader reader(in);
  reader.read_form(request_multimap);
  std::string requestID = get_from_multimap(request_multimap,"RequestID");
  if (requestID=="SaveToFile") savetoFile_ = true;  
  if (requestID=="WriteToDB") writetoDB_ = true;  
}


void SiPixelHistoricInfoWebInterface::handleEDARequest(xgi::Input* in, xgi::Output* out) {
  CgiReader reader(in);
  reader.read_form(request_multimap);
  std::string requestID = get_from_multimap(request_multimap,"RequestID");
  if (requestID=="SaveToFile") savetoFile_ = true;  
  if (requestID=="WriteToDB") writetoDB_ = true;  
}
