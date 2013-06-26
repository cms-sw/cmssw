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

#include "time.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>

#if defined(XERCES_NEW_IOSTREAMS)
#include <iostream>
#else
#include <iostream.h>
#endif

XERCES_CPP_NAMESPACE_USE
#define XML(str) XMLString::transcode(str)


class HcalDQMDbInterface{
public:
  
  HcalDQMDbInterface(){};

  DOMDocument* createDocument();  
  void writeDocument(DOMDocument* doc, const char* xmlFile);

  DOMElement* createElement(DOMDocument* doc, DOMElement* parent, const char* name);
  DOMElement* createElement(DOMDocument* doc, DOMElement* parent, const char* name, const char* value);
  DOMElement* createIOV(DOMDocument* doc, DOMElement*  parent,
			unsigned long long fIovBegin, unsigned long long fIovEnd);
  DOMElement* createTag(DOMDocument* doc, DOMElement*  parent,
			const char* fTagName, const char* fDetectorName, const char* fComment);
  DOMElement* makeMapTag(DOMDocument* doc, DOMElement* fMap);
  DOMElement* makeMapIOV(DOMDocument* doc, DOMElement* fTag);
  DOMElement* makeMapDataset(DOMDocument* doc, DOMElement* fIov);
  DOMElement* createFooter(DOMDocument* doc,
			   unsigned long long fIovBegin, unsigned long long fIovEnd,
			   const char* fTagName, const char* fDetectorName, const char* fComment);
  DOMElement* createChannel(DOMDocument* doc,DOMElement* parent, HcalDetId id);

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
  
  DOMElement* createData(DOMDocument* doc,DOMElement* parent, HcalDQMChannelQuality::Item item);
  void createDataset(DOMDocument* doc, HcalDQMChannelQuality::Item item, const char* gmtime, const char* version);
  void createHeader(DOMDocument* doc, unsigned int runno, const char* startTime);

};

class HcalHLXMaskDbInterface : public HcalDQMDbInterface {
 public:

  HcalHLXMaskDbInterface(){};

  void createData(DOMDocument* doc,DOMElement* parent, HcalHLXMask mask);
  DOMElement* createDataset(DOMDocument* doc, const HcalHLXMask mask, const char* gmtime, const char* version, const char* subversion);
  void createHeader(DOMDocument* doc);
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
