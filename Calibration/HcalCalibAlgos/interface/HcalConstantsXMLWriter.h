#ifndef _HCALCONSTANTSXMLWRITER_H
#define _HCALCONSTANTSXMLWRITER_H
#include <memory>
#include <map>
#include <vector>
// Xerces-C
#include <xercesc/util/XMLString.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMText.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOMImplementationRegistry.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMWriter.hpp>
#include <xercesc/util/XMLString.hpp>

#include <fstream>
#include <iostream>

class HcalConstantsXMLWriter
{
   public:
     HcalConstantsXMLWriter();
     virtual ~HcalConstantsXMLWriter();
     void writeXML(std::string&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&, const std::vector<int>&,const std::vector<float>&);
     
     void newCellLine(xercesc::DOMElement*, int,int,int,int,float);

   private:   
     std::string hcalfileOut_;
     xercesc::DOMImplementation* mDom;
     xercesc::DOMDocument* mDoc;
};


#endif

