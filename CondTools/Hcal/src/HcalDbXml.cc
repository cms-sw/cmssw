
//
// F.Ratnikov (UMd), Oct 28, 2005
// $Id: HcalDbXml.cc,v 1.10 2012/11/12 20:53:38 dlange Exp $
//
#include <vector>
#include <string>
#include <fstream>
#include <sstream>


#include "CalibFormats/HcalObjects/interface/HcalText2DetIdConverter.h"

#include "CondFormats/HcalObjects/interface/AllObjects.h"

// Xerces-C
#include <xercesc/util/XMLString.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMText.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOMImplementationRegistry.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMWriter.hpp>

#include "CondTools/Hcal/interface/StreamOutFormatTarget.h"

#include "CondTools/Hcal/interface/HcalDbXml.h"

using namespace std;
using namespace xercesc;

namespace {
  template <class T> XMLCh* transcode (const T& fInput) {
    ostringstream ost;
    ost << fInput;
    return XMLString::transcode (ost.str().c_str());
  } 

  const char* IOV_ID = "IOV_ID";
  const char* TAG_ID = "TAG_ID";

  std::string kind (const HcalPedestals& fObject) { return "HCAL_PEDESTALS_V2";}
  std::string kind (const HcalGains& fObject) { return "HCAL Gains";}
  std::string kind (const HcalRawGains& fObject) { return "HCAL Raw Gains";}

  std::string extensionTableName (const std::string& fKind) {
    if (fKind == "HCAL_PEDESTALS_V2") return "HCAL_PEDESTALS_V2";
    if (fKind == "HCAL Gains") return "HCAL_GAIN_PEDSTL_CALIBRATIONS";
    if (fKind == "HCAL Raw Gains") return "HCAL_RAW_GAINS";
    return "UNKNOWN";
  }
}

class XMLDocument {
public:
  XMLDocument ();
  template <class T> DOMElement* newElement (DOMElement* fParent, const T& fName);
  template <class T> DOMElement* newValue (DOMElement* fParent, const std::string& fName, const T& fValue);
  template <class T> void addAttribute (DOMElement* fElement, const std::string& fName, const T& fValue);
  const DOMDocument* document ();
  void streamOut (std::ostream& fOut);

  DOMElement* root ();
  DOMElement* makeHeader (DOMElement* fRoot, const std::string& fExtensionName, unsigned long fRun);
  DOMElement* makeDataset (DOMElement* fRoot, const std::string& fVersion);
  DOMElement* makeElement (DOMElement* fRoot);
  DOMElement* makeMaps (DOMElement* fRoot);

  DOMElement* makeType (DOMElement* fHeader, const std::string& fExtensionName);
  DOMElement* makeRun (DOMElement* fHeader, unsigned long fRun);

  DOMElement* makeChId (DOMElement* fDataset, DetId fId);

  DOMElement* makeElementDataset (DOMElement* fElement, int fXMLId, DetId fDetId, const std::string& fVersion, const std::string& fKind, unsigned long fRun);
  DOMElement* makeElementIOV (DOMElement* fElement, unsigned long long fIovBegin, unsigned long long fIovEnd = 0);
  DOMElement* makeElementTag (DOMElement* fElement, const std::string& fTagName, const std::string& fDetectorName, const std::string& fComment = "Automatically created by HcalDbXml");

  DOMElement* makeMapTag (DOMElement* fMap);
  DOMElement* makeMapIOV (DOMElement* fTag);
  DOMElement* makeMapDataset (DOMElement* fIov, int fXMLId);

  DOMElement* makeData (DOMElement* fDataset, const HcalPedestal& fPed, const HcalPedestalWidth& fWidth);
  DOMElement* makeData (DOMElement* fDataset);

  // specializations
  void addData (DOMElement* fData, const HcalPedestal& fItem);
  void addData (DOMElement* fData, const HcalPedestalWidth& fItem);
  void addData (DOMElement* fData, const HcalGain& fItem);
  void addData (DOMElement* fData, const HcalRawGain& fItem);
  void addData (DOMElement* fData, const HcalGainWidth& fItem);

private:
  DOMImplementation* mDom;
  DOMDocument* mDoc;
};

  XMLDocument::XMLDocument () 
    : mDoc (0)
  {
    XMLPlatformUtils::Initialize();
    mDom =  DOMImplementationRegistry::getDOMImplementation (transcode ("Core"));
    mDoc = mDom->createDocument(
				0,                    // root element namespace URI.
				transcode ("ROOT"),         // root element name
				0);                   // document type object (DTD).
  }

  template <class T> DOMElement* XMLDocument::newElement (DOMElement* fParent, const T& fName) {
    DOMElement* element = mDoc->createElement (transcode (fName));
    fParent->appendChild (element);
    return element;
  }

  template <class T> DOMElement* XMLDocument::newValue (DOMElement* fParent, const std::string& fName, const T& fValue) {
    DOMElement* element = newElement (fParent, fName);
    DOMText* text = mDoc->createTextNode (transcode (fValue));
    element->appendChild (text);
    return element;
  }
  
  template <class T> void XMLDocument::addAttribute (DOMElement* fElement, const std::string& fName, const T& fValue) {
    fElement->setAttribute (transcode (fName), transcode (fValue));
  }
  
  DOMElement* XMLDocument::root () { return mDoc->getDocumentElement();}

  DOMElement* XMLDocument::makeHeader (DOMElement* fRoot, const std::string& fExtensionName, unsigned long fRun) {
    DOMElement* header = newElement (fRoot, "HEADER");
    makeType (header, fExtensionName);
    makeRun (header, fRun);
    return header;
  }

  DOMElement* XMLDocument::makeType (DOMElement* fHeader, const std::string& fKind) {
      DOMElement* type = newElement (fHeader, "TYPE");
      newValue (type, "EXTENSION_TABLE_NAME", extensionTableName (fKind));
      newValue (type, "NAME", fKind);
      return type;
  }

  DOMElement* XMLDocument::makeRun (DOMElement* fHeader, unsigned long fRun) {
      DOMElement* run =newElement (fHeader, "RUN");
      newValue (run, "RUN_TYPE", "HcalDbXml");
      newValue (run, "RUN_NUMBER", fRun);
      return run;
  }

  DOMElement* XMLDocument::makeDataset (DOMElement* fRoot, const std::string& fVersion) {
      DOMElement* dataset =newElement (fRoot, "DATA_SET");
      newValue (dataset, "VERSION", fVersion);
      return dataset;
  }

  DOMElement* XMLDocument::makeChId (DOMElement* fDataset, DetId fId) {
    DOMElement* channel = newElement (fDataset, "CHANNEL");
    newValue (channel, "EXTENSION_TABLE_NAME", "HCAL_CHANNELS");
    HcalText2DetIdConverter parser (fId);
    newValue (channel, "DETECTOR_NAME", parser.getFlavor ());
    int eta = parser.getField (1);
    newValue (channel, "ETA", abs(eta));
    newValue (channel, "Z", eta > 0 ? 1 : -1);
    newValue (channel, "PHI", parser.getField2 ());
    newValue (channel, "DEPTH", parser.getField3 ());
    newValue (channel, "HCAL_CHANNEL_ID", fId.rawId());
    return channel;
  }

  DOMElement* XMLDocument::makeElementDataset (DOMElement* fElement, int fXMLId, DetId fDetId, const std::string& fVersion, const std::string& fKind, unsigned long fRun) {
    DOMElement* dataset = newElement (fElement, "DATA_SET");
    addAttribute (dataset, "id", fXMLId);
    newValue (newElement (dataset, "KIND_OF_CONDITION"), "NAME", fKind);
    newValue (dataset, "VERSION", fVersion);
    makeRun (dataset, fRun);
    makeChId (dataset, fDetId);
    return dataset;
  }

  DOMElement* XMLDocument::makeElementIOV (DOMElement* fElement, unsigned long long fIovBegin, unsigned long long fIovEnd) {
    DOMElement* iov = newElement (fElement, "IOV");
    addAttribute (iov, "id", IOV_ID);
    newValue (iov, "INTERVAL_OF_VALIDITY_BEGIN", fIovBegin);
    if (fIovEnd) {
      newValue (iov, "INTERVAL_OF_VALIDITY_END", fIovEnd);
    }
    return iov;
  }

  DOMElement* XMLDocument::makeElementTag (DOMElement* fElement, const std::string& fTagName, const std::string& fDetectorName, const std::string& fComment) {
    DOMElement* tag = newElement (fElement, "TAG");
    addAttribute (tag, "id", TAG_ID);
    addAttribute (tag, "mode", "create");
    newValue (tag, "TAG_NAME", fTagName);
    newValue (tag, "DETECTOR_NAME", fDetectorName);
    newValue (tag, "COMMENT_DESCRIPTION", fComment);
    return tag;
  }

  DOMElement* XMLDocument::makeElement (DOMElement* fRoot) {
    DOMElement* element = newElement (fRoot, "ELEMENTS");
    return element;
  }
  
  DOMElement* XMLDocument::makeMaps (DOMElement* fRoot) {
    DOMElement* map = newElement (fRoot, "MAPS");
    return map;
  }

  DOMElement* XMLDocument::makeMapTag (DOMElement* fMap) {
    DOMElement* tag = newElement (fMap, "TAG");
    addAttribute (tag, "idref", TAG_ID);
    return tag;
  }

  DOMElement* XMLDocument::makeMapIOV (DOMElement* fTag) {
    DOMElement* iov = newElement (fTag, "IOV");
    addAttribute (iov, "idref", IOV_ID);
    return iov;
  }

  DOMElement* XMLDocument::makeMapDataset (DOMElement* fIov, int fXMLId) {
    DOMElement* element = newElement (fIov, "DATA_SET");
    addAttribute (element, "idref", fXMLId);
    return element;
  }
  
  DOMElement* XMLDocument::makeData (DOMElement* fDataset) {
    return  newElement (fDataset, "DATA");
  }

  void XMLDocument::addData (DOMElement* fData, const HcalPedestal& fItem) {
    newValue (fData, "CAPACITOR_0_VALUE", fItem.getValue (0));
    newValue (fData, "CAPACITOR_1_VALUE", fItem.getValue (1));
    newValue (fData, "CAPACITOR_2_VALUE", fItem.getValue (2));
    newValue (fData, "CAPACITOR_3_VALUE", fItem.getValue (3));
  }

  void XMLDocument::addData (DOMElement* fData, const HcalPedestalWidth& fItem) {
    // widths
    newValue (fData, "SIGMA_0_0", fItem.getSigma (0, 0));
    newValue (fData, "SIGMA_1_1", fItem.getSigma (1, 1));
    newValue (fData, "SIGMA_2_2", fItem.getSigma (2, 2));
    newValue (fData, "SIGMA_3_3", fItem.getSigma (3, 3));
    newValue (fData, "SIGMA_0_1", fItem.getSigma (0, 1));
    newValue (fData, "SIGMA_0_2", fItem.getSigma (0, 2));
    newValue (fData, "SIGMA_0_3", fItem.getSigma (0, 3));
    newValue (fData, "SIGMA_1_2", fItem.getSigma (1, 2));
    newValue (fData, "SIGMA_1_3", fItem.getSigma (1, 3));
    newValue (fData, "SIGMA_2_3", fItem.getSigma (2, 3));
  }

  void XMLDocument::addData (DOMElement* fData, const HcalGain& fItem) {
    newValue (fData, "CAPACITOR_0_VALUE", fItem.getValue (0));
    newValue (fData, "CAPACITOR_1_VALUE", fItem.getValue (1));
    newValue (fData, "CAPACITOR_2_VALUE", fItem.getValue (2));
    newValue (fData, "CAPACITOR_3_VALUE", fItem.getValue (3));
  }

  void XMLDocument::addData (DOMElement* fData, const HcalRawGain& fItem) {
    newValue (fData, "VALUE", fItem.getValue ());
    newValue (fData, "ERROR", fItem.getError ());
    newValue (fData, "VOLTAGE", fItem.getVoltage ());
    newValue (fData, "STATUS", fItem.strStatus ());
  }

 void XMLDocument::addData (DOMElement* fData, const HcalGainWidth& fItem) {
    newValue (fData, "CAPACITOR_0_ERROR", fItem.getValue (0));
    newValue (fData, "CAPACITOR_1_ERROR", fItem.getValue (1));
    newValue (fData, "CAPACITOR_2_ERROR", fItem.getValue (2));
    newValue (fData, "CAPACITOR_3_ERROR", fItem.getValue (3));
  }


  DOMElement* XMLDocument::makeData (DOMElement* fDataset, const HcalPedestal& fPed, const HcalPedestalWidth& fWidth) {
    DOMElement* data = newElement (fDataset, "DATA");
    // pedestals
    newValue (data, "CAPACITOR_0_VALUE", fPed.getValue (0));
    newValue (data, "CAPACITOR_1_VALUE", fPed.getValue (1));
    newValue (data, "CAPACITOR_2_VALUE", fPed.getValue (2));
    newValue (data, "CAPACITOR_3_VALUE", fPed.getValue (3));
    // widths
    newValue (data, "SIGMA_0_0", fWidth.getSigma (0, 0));
    newValue (data, "SIGMA_1_1", fWidth.getSigma (1, 1));
    newValue (data, "SIGMA_2_2", fWidth.getSigma (2, 2));
    newValue (data, "SIGMA_3_3", fWidth.getSigma (3, 3));
    newValue (data, "SIGMA_0_1", fWidth.getSigma (0, 1));
    newValue (data, "SIGMA_0_2", fWidth.getSigma (0, 2));
    newValue (data, "SIGMA_0_3", fWidth.getSigma (0, 3));
    newValue (data, "SIGMA_1_2", fWidth.getSigma (1, 2));
    newValue (data, "SIGMA_1_3", fWidth.getSigma (1, 3));
    newValue (data, "SIGMA_2_3", fWidth.getSigma (2, 3));
    return data;
  }


  const DOMDocument* XMLDocument::document () {return mDoc;}

  void XMLDocument::streamOut (std::ostream& fOut) {
    StreamOutFormatTarget formTaget (fOut);
    DOMWriter* domWriter = mDom->createDOMWriter();
    domWriter->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);
    domWriter->writeNode (&formTaget, *(root()));
    mDoc->release ();
  }


template <class T1, class T2>
bool dumpObject_ (std::ostream& fOutput, 
		  unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, 
		  const T1* fObject1, const T2* fObject2 = 0) {
  if (!fObject1) return false;
  const std::string KIND = kind (*fObject1);
  
  XMLDocument doc;
  DOMElement* root = doc.root ();
  doc.makeHeader (root, KIND, fRun);

  DOMElement* elements = doc.makeElement (root);
  doc.makeElementIOV (elements, fGMTIOVBegin, fGMTIOVEnd);
  doc.makeElementTag (elements, fTag, "HCAL");

  DOMElement* iovmap = doc.makeMapIOV (doc.makeMapTag (doc.makeMaps (root)));
  
  std::vector<DetId> detids = fObject1->getAllChannels ();
  for (unsigned iCh = 0; iCh < detids.size(); iCh++) {
    DetId id = detids [iCh];
    ostringstream version;
    version << fTag << '_' << fGMTIOVBegin; // CONVENTION: version == tag + iov for initial setting
    DOMElement* dataset = doc.makeDataset (root, version.str());  
    doc.makeChId (dataset, id);
    DOMElement* data = doc.makeData (dataset);
    doc.addData (data, *(fObject1->getValues (id)));
    try {
      if (fObject2) doc.addData (data, *(fObject2->getValues (id)));
    }
    catch (...) {
      std::cout << "dumpObject_-> ERROR: width is not available for cell # " << id.rawId() << std::endl;
    }
    doc.makeElementDataset (elements, iCh, id, version.str(), KIND, fRun);
    doc.makeMapDataset (iovmap, iCh);
  }
  doc.streamOut (fOutput);
  return true;
}


bool HcalDbXml::dumpObject (std::ostream& fOutput, 
			    unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, 
			    const HcalPedestals& fObject, const HcalPedestalWidths& fError) {
  return dumpObject_ (fOutput, fRun, fGMTIOVBegin, fGMTIOVEnd, fTag, &fObject, &fError);
}

bool HcalDbXml::dumpObject (std::ostream& fOutput, 
			    unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, 
			    const HcalPedestals& fObject) {
  float dummyError = 0.0001;
  std::cout << "HcalDbXml::dumpObject-> set default errors: 0.0001, 0.0001, 0.0001, 0.0001" << std::endl;
  HcalPedestalWidths widths(fObject.topo(), fObject.isADC() );
  std::vector<DetId> channels = fObject.getAllChannels ();
  for (std::vector<DetId>::iterator channel = channels.begin ();
       channel !=  channels.end ();
       channel++) {

    HcalPedestalWidth item(*channel);
    for (int iCapId = 0; iCapId < 4; iCapId++) {
      item.setSigma (iCapId, iCapId, dummyError*dummyError);
    }
    widths.addValues(item);
  }
  return dumpObject (fOutput, fRun, fGMTIOVBegin, fGMTIOVEnd, fTag, fObject, widths);
}

bool HcalDbXml::dumpObject (std::ostream& fOutput, 
			    unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, 
			    const HcalGains& fObject, const HcalGainWidths& fError) {
  return dumpObject_ (fOutput, fRun, fGMTIOVBegin, fGMTIOVEnd, fTag, &fObject, &fError);
}

bool HcalDbXml::dumpObject (std::ostream& fOutput, 
			    unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, 
			    const HcalGains& fObject) {
  HcalGainWidths widths(fObject.topo());
  std::vector<DetId> channels = fObject.getAllChannels ();
  for (std::vector<DetId>::iterator channel = channels.begin (); channel !=  channels.end (); channel++) 
    {
      HcalGainWidth item(*channel,0,0,0,0);
      widths.addValues(item); // no error
    }
  return dumpObject (fOutput, fRun, fGMTIOVBegin, fGMTIOVEnd, fTag, fObject, widths);
}

bool HcalDbXml::dumpObject (std::ostream& fOutput, 
			    unsigned fRun, unsigned long fGMTIOVBegin, unsigned long fGMTIOVEnd, const std::string& fTag, 
			    const HcalRawGains& fObject) {
  return dumpObject_ (fOutput, fRun, fGMTIOVBegin, fGMTIOVEnd, fTag, &fObject, (const HcalGainWidths*)0);
}
