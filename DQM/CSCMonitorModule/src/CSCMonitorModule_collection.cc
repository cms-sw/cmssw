/*  =====================================================================================
 *
 *       Filename:  CSCMonitorModule_collection.cc
 *
 *    Description:  Method loadXMLBookingInfo implementation
 *
 *        Version:  1.0
 *        Created:  04/18/2008 03:39:49 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Valdas Rapsevicius (VR), Valdas.Rapsevicius@cern.ch
 *        Company:  CERN, CH
 *
 *  =====================================================================================
 */

#include <DQM/CSCMonitorModule/interface/CSCMonitorModule.h>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#include "csc_utilities.cc"

#define DEF_HISTO_COLOR 48

using namespace XERCES_CPP_NAMESPACE;


/**
 * @brief  Get MonitorElement by using path name
 * @param  name Path to the histogram (MonitorElement)
 * @param  me Pointer to resulting MonitorElement 
 * @return true if histogram found and pointer value changed, false - otherwise
 */
const bool CSCMonitorModule::isMEValid(const std::string name, MonitorElement*& me) {
  me = dbe->get(name);
  if(me == NULL) {
    LOGWARNING("ME not found") << "MonitorElement [" << name << "] not found.";
    return false;
  } else {
    return true;
  }
}

/**
 * @brief  Checks if MonitorElement of EventInfo level is available and returns it
 * @param  name Object name
 * @param  me Pointer to the Object to be returned
 * @return true if object was found and false otherwise
 */
const bool CSCMonitorModule::MEEventInfo(const std::string name, MonitorElement*& me) {
  return isMEValid(rootDir + EVENTINFO_FOLDER + name, me);
}

/**
 * @brief  Checks if MonitorElement of EventInfo/reportSummaryContents level is available and returns it
 * @param  name Object name
 * @param  me Pointer to the Object to be returned
 * @return true if object was found and false otherwise
 */
const bool CSCMonitorModule::MEReportSummaryContents(const std::string name, MonitorElement*& me) {
  return isMEValid(rootDir + SUMCONTENTS_FOLDER + name, me);
}

/**
 * @brief  Checks if MonitorElement of EMU level is available and returns it
 * @param  name Histogram name
 * @param  me Pointer to the histogram to be returned
 * @return true if histogram was found and false otherwise
 */
const bool CSCMonitorModule::MEEMU(const std::string name, MonitorElement*& me) {
  return isMEValid(rootDir + SUMMARY_FOLDER + name, me);
}


/**
 * @brief  Checks if MonitorElement of DDU level is available and returns it
 * @param  dduId DDU number
 * @param  name Histogram name
 * @param  me Pointer to the histogram to be returned
 * @return true if histogram was found and false otherwise
 */
const bool CSCMonitorModule::MEDDU(const unsigned int dduId, const std::string name, MonitorElement*& me) {

  std::string buffer;

  bool result = isMEValid(rootDir + DDU_FOLDER + getDDUTag(dduId, buffer) + "/" + name, me);
  if (!result && hitBookDDU) {
    LOGINFO("DDU ME booking on demand") << "DDU id = " << dduId << " is being booked on demand (hitBookDDU = " << std::boolalpha << hitBookDDU << ")";
    dbe->setCurrentFolder(rootDir + DDU_FOLDER + getDDUTag(dduId, buffer));
    book("DDU");
    result = isMEValid(rootDir + DDU_FOLDER + getDDUTag(dduId, buffer) + "/" + name, me);
  }

  return result;

}


/**
 * @brief  Load XML file and create definitions
 * @param  
 * @return 
 */
int CSCMonitorModule::loadCollection() {

  XMLPlatformUtils::Initialize();
  XercesDOMParser *parser = new XercesDOMParser();
  parser->setValidationScheme(XercesDOMParser::Val_Always);
  parser->setDoNamespaces(true);
  parser->setDoSchema(true);
  parser->setValidationSchemaFullChecking(false); // this is default
  parser->setCreateEntityReferenceNodes(true);  // this is default
  parser->setIncludeIgnorableWhitespace (false);

  parser->parse(bookingFile.c_str());
  DOMDocument *doc = parser->getDocument();
  DOMNode *docNode = (DOMNode*) doc->getDocumentElement();
  
  std::string nodeName = XMLString::transcode(docNode->getNodeName());
  if( nodeName != "Booking" ){
    LOGERROR("loadCollection") << "Wrong booking root node: " << XMLString::transcode(docNode->getNodeName());
    delete parser;
    return 1;
  }
  DOMNodeList *itemList = docNode->getChildNodes();

  for(uint32_t i=0; i < itemList->getLength(); i++) {

    nodeName = XMLString::transcode(itemList->item(i)->getNodeName());
    if(nodeName != "Histogram") {
      continue;
    }

    DOMNodeList *props  = itemList->item(i)->getChildNodes();
    Histo h;
    std::string prefix = "", name = "";
    for(uint32_t j = 0; j < props->getLength(); j++) {
      std::string tname  = XMLString::transcode(props->item(j)->getNodeName());
      std::string tvalue = XMLString::transcode(props->item(j)->getTextContent());
      h.insert(std::make_pair(tname, tvalue));
      if(tname == "Name")   name   = tvalue;
      if(tname == "Prefix") prefix = tvalue;
    }

    if(!name.empty() && !prefix.empty()) {
      HistoDefMapIter it = collection.find(prefix);
      if( it == collection.end()) {
        HistoDef hd;
        hd.insert(make_pair(name, h));
        collection.insert(make_pair(prefix, hd)); 
      } else {
        it->second.insert(make_pair(name, h));
      }
    }

  }

  delete parser;

  std::ostringstream buffer;
  buffer << std::endl;
  for(HistoDefMapIter hdmi = collection.begin(); hdmi != collection.end(); hdmi++) {
    buffer << " # of " << hdmi->first << " histograms loaded =  " << hdmi->second.size() << std::endl;
  }
  LOGINFO("Histograms loaded") << buffer.str();

  return 0;
}


/**
 * @brief  Book a group of histograms
 * @param  prefix name of histogram group to book
 * @return 
 */
void CSCMonitorModule::book(const std::string prefix) {

  HistoDefMapIter hdmi = collection.find(prefix);

  if( hdmi != collection.end()) {

    for(HistoDefIter hdi = hdmi->second.begin(); hdi != hdmi->second.end(); hdi++) {
      
      MonitorElement* me = NULL;
      std::string name, type, title, s;
      int i, j, l;
      double d, e, f, g, h, k;
      
      name  = hdi->first;
      type  = getHistoValue(hdi->second, "Type", type, "h1");
      title = getHistoValue(hdi->second, "Title", title, hdi->first);

      if (type == "h1") {
        me = dbe->book1D(name, title,
            getHistoValue(hdi->second, "XBins", i, 1),
            getHistoValue(hdi->second, "XMin",  d, 0),
            getHistoValue(hdi->second, "XMax",  e, 1));
      }
      if(type == "h2") {
        me = dbe->book2D(name, title,
            getHistoValue(hdi->second, "XBins", i, 1),
            getHistoValue(hdi->second, "XMin",  d, 0),
            getHistoValue(hdi->second, "XMax",  e, 1),
            getHistoValue(hdi->second, "YBins", j, 1),
            getHistoValue(hdi->second, "YMin",  f, 0),
            getHistoValue(hdi->second, "YMax",  g, 1));
      }
      if(type == "h3") {
        me = dbe->book3D(name, title,
            getHistoValue(hdi->second, "XBins", i, 1),
            getHistoValue(hdi->second, "XMin",  d, 0),
            getHistoValue(hdi->second, "XMax",  e, 1),
            getHistoValue(hdi->second, "YBins", j, 1),
            getHistoValue(hdi->second, "YMin",  f, 0),
            getHistoValue(hdi->second, "YMax",  g, 1),
            getHistoValue(hdi->second, "ZBins", l, 1),
            getHistoValue(hdi->second, "ZMin",  h, 0),
            getHistoValue(hdi->second, "ZMax",  k, 1));
      }
      if(type == "hp") {
        me = dbe->bookProfile(name, title,
            getHistoValue(hdi->second, "XBins", i, 1),
            getHistoValue(hdi->second, "XMin",  d, 0),
            getHistoValue(hdi->second, "XMax",  e, 1),
            getHistoValue(hdi->second, "YBins", j, 1),
            getHistoValue(hdi->second, "YMin",  f, 0),
            getHistoValue(hdi->second, "YMax",  g, 1));
      }
      if(type == "hp2") {
        me = dbe->bookProfile2D(name, title,
            getHistoValue(hdi->second, "XBins", i, 1),
            getHistoValue(hdi->second, "XMin",  d, 0),
            getHistoValue(hdi->second, "XMax",  e, 1),
            getHistoValue(hdi->second, "YBins", j, 1),
            getHistoValue(hdi->second, "YMin",  f, 0),
            getHistoValue(hdi->second, "YMax",  g, 1),
            getHistoValue(hdi->second, "ZBins", l, 1),
            getHistoValue(hdi->second, "ZMin",  h, 0),
            getHistoValue(hdi->second, "ZMax",  k, 1));
      }

      if(me != NULL) {
        TH1 *h = me->getTH1();
        if(findHistoValue(hdi->second, "XTitle", s)) me->setAxisTitle(s, 1);
        if(findHistoValue(hdi->second, "YTitle", s)) me->setAxisTitle(s, 2);
        if(findHistoValue(hdi->second, "ZTitle", s)) me->setAxisTitle(s, 3);
        if(findHistoValue(hdi->second, "SetOption", s)) h->SetOption(s.c_str());
        if(findHistoValue(hdi->second, "SetiStats", i)) h->SetStats(i);
        h->SetFillColor(getHistoValue(hdi->second, "SetFillColor", i, DEF_HISTO_COLOR));
        if(findHistoValue(hdi->second, "SetXLabels", s)) {
          std::map<int, std::string> labels;
          ParseAxisLabels(s, labels);
          for (std::map<int, std::string>::iterator l_itr = labels.begin(); l_itr != labels.end(); ++l_itr) {
            h->GetXaxis()->SetBinLabel(l_itr->first, l_itr->second.c_str());
          }
        }
        if(findHistoValue(hdi->second, "SetYLabels", s)) {
          std::map<int, std::string> labels;
          ParseAxisLabels(s, labels);
          for (std::map<int, std::string>::iterator l_itr = labels.begin(); l_itr != labels.end(); ++l_itr) {
            h->GetYaxis()->SetBinLabel(l_itr->first, l_itr->second.c_str());
          }
        }
        if(findHistoValue(hdi->second, "LabelOption", s)) {
          std::vector<std::string> v;
          if(2 == tokenize(s, v, ",")) {
            h->LabelsOption(v[0].c_str(), v[1].c_str());
          }
        }
        if(findHistoValue(hdi->second, "SetLabelSize", s)) {
          std::vector<std::string> v;
          if(2 == tokenize(s, v, ",")) {
            h->SetLabelSize((double) atof(v[0].c_str()), v[1].c_str());
          }
        }
        if(findHistoValue(hdi->second, "SetTitleOffset", s)) {
          std::vector<std::string> v;
          if(2 == tokenize(s, v, ",")) {
            h->SetTitleOffset((double) atof(v[0].c_str()), v[1].c_str());
          }
        }
        if(findHistoValue(hdi->second, "SetMinimum", d)) h->SetMinimum(d);
        if(findHistoValue(hdi->second, "SetMaximum", d)) h->SetMaximum(d);
        if(findHistoValue(hdi->second, "SetNdivisionsX", i)) {
          h->SetNdivisions(i, "X");
          h->GetXaxis()->CenterLabels(true);
        }
        if(findHistoValue(hdi->second, "SetNdivisionsY", i)) {
          h->SetNdivisions(i, "Y");
          h->GetYaxis()->CenterLabels(true);
        }
        if(findHistoValue(hdi->second, "SetTickLengthX", d)) h->SetTickLength(d, "X");
        if(findHistoValue(hdi->second, "SetTickLengthY", d)) h->SetTickLength(d, "Y");
        if(findHistoValue(hdi->second, "SetLabelSizeX", d)) h->SetLabelSize(d, "X");
        if(findHistoValue(hdi->second, "SetLabelSizeY", d)) h->SetLabelSize(d, "Y");
        if(findHistoValue(hdi->second, "SetLabelSizeZ", d)) h->SetLabelSize(d, "Z");
        if(findHistoValue(hdi->second, "SetErrorOption", s)) reinterpret_cast<TProfile*>(h)->SetErrorOption(s.c_str());

      }

    }
  }
}

/**
 * @brief  Print collection of available histograms and their parameters
 * @param  
 * @return 
 */
void CSCMonitorModule::printCollection(){

  std::ostringstream buffer;
  for(HistoDefMapIter hdmi = collection.begin(); hdmi != collection.end(); hdmi++) {
    buffer << hdmi->first << " [" << std::endl;
    for(HistoDefIter hdi = hdmi->second.begin(); hdi != hdmi->second.end(); hdi++) {
      buffer << "   " << hdi->first << " [" << std::endl;
      for(HistoIter hi = hdi->second.begin(); hi != hdi->second.end(); hi++) {
        buffer << "     " << hi->first << " = " << hi->second << std::endl;
      }
      buffer << "   ]" << std::endl;
    }
    buffer << " ]" << std::endl;
  }
  LOGINFO("Histogram collection") << buffer.str();

}

