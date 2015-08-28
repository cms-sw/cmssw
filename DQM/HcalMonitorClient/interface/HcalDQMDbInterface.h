#include <xercesc/util/XMLString.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMImplementationLS.hpp>
#include <xercesc/dom/DOMWriter.hpp>

#include "DQM/HcalMonitorClient/interface/HcalDQMChannelQuality.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "CalibFormats/HcalObjects/interface/HcalText2DetIdConverter.h"
#include "DQM/HcalMonitorClient/interface/HcalHLXMask.h"
//#include "CondTools/Hcal/include/HcalDbXml.h"

#include <ctime>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

#if defined(XERCES_NEW_IOSTREAMS)
#include <iostream>
#else
#include <iostream.h>
#endif

#define XML(str) XMLString::transcode(str)


class HcalDQMDbInterface{
public:
  
  HcalDQMDbInterface(){};

  XERCES_CPP_NAMESPACE::DOMDocument* createDocument();  
  void writeDocument(XERCES_CPP_NAMESPACE::DOMDocument* doc, const char* xmlFile);

  XERCES_CPP_NAMESPACE::DOMElement* createElement(XERCES_CPP_NAMESPACE::DOMDocument* doc, XERCES_CPP_NAMESPACE::DOMElement* parent, const char* name);
  XERCES_CPP_NAMESPACE::DOMElement* createElement(XERCES_CPP_NAMESPACE::DOMDocument* doc, XERCES_CPP_NAMESPACE::DOMElement* parent, const char* name, const char* value);
  XERCES_CPP_NAMESPACE::DOMElement* createIOV(XERCES_CPP_NAMESPACE::DOMDocument* doc, XERCES_CPP_NAMESPACE::DOMElement*  parent,
			unsigned long long fIovBegin, unsigned long long fIovEnd);
  XERCES_CPP_NAMESPACE::DOMElement* createTag(XERCES_CPP_NAMESPACE::DOMDocument* doc, XERCES_CPP_NAMESPACE::DOMElement*  parent,
			const char* fTagName, const char* fDetectorName, const char* fComment);
  XERCES_CPP_NAMESPACE::DOMElement* makeMapTag(XERCES_CPP_NAMESPACE::DOMDocument* doc, XERCES_CPP_NAMESPACE::DOMElement* fMap);
  XERCES_CPP_NAMESPACE::DOMElement* makeMapIOV(XERCES_CPP_NAMESPACE::DOMDocument* doc, XERCES_CPP_NAMESPACE::DOMElement* fTag);
  XERCES_CPP_NAMESPACE::DOMElement* makeMapDataset(XERCES_CPP_NAMESPACE::DOMDocument* doc, XERCES_CPP_NAMESPACE::DOMElement* fIov);
  XERCES_CPP_NAMESPACE::DOMElement* createFooter(XERCES_CPP_NAMESPACE::DOMDocument* doc,
			   unsigned long long fIovBegin, unsigned long long fIovEnd,
			   const char* fTagName, const char* fDetectorName, const char* fComment);
  XERCES_CPP_NAMESPACE::DOMElement* createChannel(XERCES_CPP_NAMESPACE::DOMDocument* doc,XERCES_CPP_NAMESPACE::DOMElement* parent, HcalDetId id);

  const char* itoa(int i){
    char temp[256];
    sprintf(temp,"%d",i);
    std::string outVal(temp);
    return outVal.c_str();
  }
};


class HcalHotCellDbInterface : public HcalDQMDbInterface {
 public:
  
  HcalHotCellDbInterface(){};
  
  XERCES_CPP_NAMESPACE::DOMElement* createData(XERCES_CPP_NAMESPACE::DOMDocument* doc,XERCES_CPP_NAMESPACE::DOMElement* parent, const HcalDQMChannelQuality::Item& item);
  void createDataset(XERCES_CPP_NAMESPACE::DOMDocument* doc, const HcalDQMChannelQuality::Item& item, const char* gmtime, const char* version);
  void createHeader(XERCES_CPP_NAMESPACE::DOMDocument* doc, unsigned int runno, const char* startTime);

};

class HcalHLXMaskDbInterface : public HcalDQMDbInterface {
 public:

  HcalHLXMaskDbInterface(){};

  void createData(XERCES_CPP_NAMESPACE::DOMDocument* doc,XERCES_CPP_NAMESPACE::DOMElement* parent, const HcalHLXMask& mask);
  XERCES_CPP_NAMESPACE::DOMElement* createDataset(XERCES_CPP_NAMESPACE::DOMDocument* doc, const HcalHLXMask& mask, const char* gmtime, const char* version, const char* subversion);
  void createHeader(XERCES_CPP_NAMESPACE::DOMDocument* doc);
};




/*
class HcalRunSummaryDbInterface : public HcalDQMDbInterface{
 public:
  
  HcalDQMDbInterface(){};
  
  DOMElement* createChannel(DOMDocument* doc,DOMElement* parent );
  DOMElement* createData(DOMDocument* doc,DOMElement* parent );
  void createDataset(DOMDocument* doc);
  
  
};
*/
