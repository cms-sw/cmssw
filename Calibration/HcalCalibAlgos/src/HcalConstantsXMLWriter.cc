#include "Calibration/HcalCalibAlgos/interface/HcalConstantsXMLWriter.h"

// Write the new XML object: needed includes

// Xerces-C
#include <xercesc/dom/DOMNode.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMCharacterData.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include "FWCore/Concurrency/interface/Xerces.h"
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/util/XMLURL.hpp>


#include "CondTools/Hcal/interface/StreamOutFormatTarget.h"
#include <sstream>
#include <string>
using namespace std;
using namespace xercesc;

HcalConstantsXMLWriter::HcalConstantsXMLWriter()
{
}
HcalConstantsXMLWriter::~HcalConstantsXMLWriter()
{
}
void HcalConstantsXMLWriter::writeXML(string& newfile0,const vector<int>& detvec,const vector<int>& etavec,const vector<int>& phivec,const vector<int>& depthvec,const vector<float>& scalevec)
{
   int nn = newfile0.size();
   char newfile[99]; 
   for (int i = 0; i < nn; i++)
   {
     newfile[i]=newfile0[i];
   }
   char const* fend="\0";
   newfile[nn] = *fend;

   cout<<" New file "<<newfile<<endl;
 
   filebuf fb;
   fb.open (newfile,ios::out);
   ostream fOut(&fb);
   
   XMLCh tempStr[100];
   
   XMLString::transcode ("Core",tempStr,99);
   mDom = DOMImplementationRegistry::getDOMImplementation (tempStr);

   XMLString::transcode("CalibrationConstants", tempStr, 99);
   mDoc = mDom->createDocument(
                                0,                    // root element namespace URI.
                                tempStr,         // root element name
                                0);                   // document type object (DTD).

   StreamOutFormatTarget formTarget (fOut);
   DOMWriter* domWriter = mDom->createDOMWriter();
   domWriter->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);
   DOMElement*   root = mDoc->getDocumentElement();

   XMLString::transcode("Hcal", tempStr, 99);
   DOMElement*   rootelem = mDoc->createElement (tempStr);
   root->appendChild(rootelem);

   XMLString::transcode("Cell", tempStr, 99);
   vector<DOMElement*> theDOMVec;

   for(unsigned int i=0; i<detvec.size();i++)
   {
   theDOMVec.push_back(mDoc->createElement (tempStr));
   newCellLine(theDOMVec[i],detvec[i],etavec[i],phivec[i],depthvec[i],scalevec[i]);
   rootelem->appendChild(theDOMVec[i]);
   }
 
   cout<<" Write Doc "<<theDOMVec.size()<<endl;
   domWriter->writeNode (&formTarget, *mDoc);
   cout<<" End of Writting "<<endl;
   mDoc->release ();
}

void HcalConstantsXMLWriter::newCellLine(DOMElement* detelem, int det, int eta, int phi, int depth,  float scale)
{
   XMLCh tempStr[100]; 
   XMLString::transcode("det_index", tempStr, 99);
   DOMAttr* attrdet = mDoc->createAttribute(tempStr);

   XMLString::transcode("eta_index", tempStr, 99);
   DOMAttr* attreta = mDoc->createAttribute(tempStr);
 
   XMLString::transcode("phi_index", tempStr, 99);
   DOMAttr* attrphi = mDoc->createAttribute(tempStr) ;
   
   XMLString::transcode("depth_index", tempStr, 99);
   DOMAttr* attrdepth =  mDoc->createAttribute(tempStr) ;

   XMLString::transcode("scale_factor", tempStr, 99);
   DOMAttr* attrscale =  mDoc->createAttribute(tempStr) ; 
 
   ostringstream ost;
   ost <<det;
   attrdet->setValue(XMLString::transcode(ost.str().c_str()));
   detelem->setAttributeNode(attrdet);

   ostringstream ost1;
   ost1 <<eta;
   attreta->setValue(XMLString::transcode(ost1.str().c_str()));
   //DOMAttr* attr3 = detelem->setAttributeNode(attreta);
   detelem->setAttributeNode(attreta);

   ostringstream ost2;
   ost2 <<phi;
   attrphi->setValue(XMLString::transcode(ost2.str().c_str()));
   //DOMAttr* attr4 = detelem->setAttributeNode(attrphi);
   detelem->setAttributeNode(attrphi);

   ostringstream ost3;
   ost3 <<depth;     
   attrdepth->setValue(XMLString::transcode(ost3.str().c_str()));
   //DOMAttr* attr5 = detelem->setAttributeNode(attrdepth);
   detelem->setAttributeNode(attrdepth);

   ostringstream ost4;
   ost4 << scale;
   attrscale->setValue(XMLString::transcode(ost4.str().c_str()));
   //DOMAttr* attr6 = detelem->setAttributeNode(attrscale);
   detelem->setAttributeNode(attrscale);

}


