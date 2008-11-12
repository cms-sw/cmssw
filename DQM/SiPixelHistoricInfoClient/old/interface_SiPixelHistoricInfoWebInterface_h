#ifndef SiPixelHistoricInfoClient_WebInterface_h
#define SiPixelHistoricInfoClient_WebInterface_h

		
#include "DQMServices/WebComponents/interface/WebInterface.h"


class SiPixelHistoricInfoWebInterface : public WebInterface {
public:
  SiPixelHistoricInfoWebInterface(std::string theContextURL, 
                                  std::string theApplicationURL, 
				  DQMOldReceiver** _mui_p);
 ~SiPixelHistoricInfoWebInterface();

  void handleCustomRequest(xgi::Input* in, xgi::Output* out) throw (xgi::exception::Exception);
  void handleEDARequest(xgi::Input* in, xgi::Output* out); 

  bool getSaveToFile() const { return savetoFile_; };
  void setSaveToFile(bool flag) { savetoFile_ = flag; };

  bool getWriteToDB() const { return writetoDB_; };
  void setWriteToDB(bool flag) { writetoDB_ = flag; };

private: 
  bool savetoFile_; 
  bool writetoDB_; 
  std::multimap<std::string, std::string> request_multimap;
};


#endif
