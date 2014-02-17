// -*- C++ -*-
//
// Package:     RPCTrigger
// Class  :     MuonsGrabber
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  
//         Created:  Thu Sep 17 14:21:01 CEST 2009
// $Id: MuonsGrabber.cc,v 1.2 2009/11/04 13:31:59 fruboes Exp $
//

// system include files

// user include files
#include "L1Trigger/RPCTrigger/interface/MuonsGrabber.h"
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <sstream>
#include <algorithm>

XERCES_CPP_NAMESPACE_USE

class XStr
{
public :
    XStr(const char* const toTranscode)
    {
        fUnicodeForm = XMLString::transcode(toTranscode);
    }

    ~XStr()
    {
        XMLString::release(&fUnicodeForm);
    }

    const XMLCh* unicodeForm() const
    {
        return fUnicodeForm;
    }

private :
    XMLCh*   fUnicodeForm;
};

#define X(str) XStr(str).unicodeForm()

//
// constants, enums and typedefs
//

//
// static data member definitions
//

MuonsGrabber & MuonsGrabber::Instance(){

	static MuonsGrabber grabber;
	return grabber;

	
}


//
// constructors and destructor
//
MuonsGrabber::MuonsGrabber()
{

    try {
       XMLPlatformUtils::Initialize();
    }
    catch(const XMLException &toCatch)  {
      throw std::string("Error during Xerces-c Initialization: "
                 + std::string(XMLString::transcode(toCatch.getMessage())));
    }
 
   m_dom = DOMImplementationRegistry::getDOMImplementation(X("Core"));
   if (m_dom == 0) throw cms::Exception("RPCMuonsGrabber") << "Cannot get DOM" << std::endl;

   m_doc = m_dom->createDocument(
                          0,                    // root element namespace URI.
                          X("rpctDataStream"),         // root element name
                          0);                   // document type object (DTD).

   m_rootElem = m_doc->getDocumentElement();
         
   m_currEvent = 0;
}


// MuonsGrabber::MuonsGrabber(const MuonsGrabber& rhs)
// {
//    // do actual copying here;
// }

MuonsGrabber::~MuonsGrabber()
{

  // save xmlfile
  XMLCh tempStr[100];
  XMLString::transcode("LS", tempStr, 99);
  DOMImplementation *impl          = DOMImplementationRegistry::getDOMImplementation(tempStr);
  DOMWriter         *theSerializer = ((DOMImplementationLS*)impl)->createDOMWriter();
  
  theSerializer->setEncoding(X("UTF-8"));

  if (theSerializer->canSetFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true))
              theSerializer->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);

  XMLFormatTarget *myFormTarget = new LocalFileFormatTarget(X("testpulses.xml"));
  DOMNode* xmlstylesheet  = m_doc->createProcessingInstruction(X("xml-stylesheet"),
                                     X("type=\"text/xsl\"href=\"default.xsl\""));

  m_doc->insertBefore(xmlstylesheet, m_rootElem);
  theSerializer->writeNode(myFormTarget, *m_doc);
        
  delete theSerializer;
  delete myFormTarget;
  m_doc->release();
      

       

}

void MuonsGrabber::startNewEvent(int event, int bx) {
	
     //<event bx="0" num="0">
      m_currEvent = m_doc->createElement(X("event"));
      m_currEvent->setAttribute(X("num"), X( IntToString(event).c_str())); 
      m_currEvent->setAttribute(X("bx"),  X( IntToString(bx).c_str())); 
      m_rootElem->appendChild(m_currEvent);  

      
      m_currentEvent = event;
      m_currentBX = bx;
}


void MuonsGrabber::addMuon(RPCTBMuon & mu, int lvl, int region, int hs, int index){
	
	if (mu.getPtCode()>0) m_muons.push_back( RPCMuonExtraStruct(lvl, region, hs, index, mu) );
}
	       
void MuonsGrabber::writeDataForRelativeBX(int bx){

  if (m_muons.empty()) return; 

  //<bxData num="11">
  DOMElement* currRelBx = m_doc->createElement(X("bxData"));
  currRelBx->setAttribute(X("num"), X( IntToString(bx).c_str()));
  m_currEvent->appendChild(currRelBx);
  
  //std::cout << "Writing out for relBx "	<< bx << std::endl;
 
  //   <hs num="1" be="0">
  // levels
  // 0 - PAC
  // 1 - tbgs 
  // 2 - tcgs
  // 3 - hs
  // 4 fs

  std::sort(m_muons.begin(), m_muons.end(), RPCMuonExtraStruct::lvlCompare) ;
  for (int tcNum = 0; tcNum <= 11; ++tcNum  ) {
    DOMElement* tc = 0;
    DOMElement* tcgs = 0;
    for (int tbNum = 0; tbNum <= 10; ++tbNum  ) { // check actual range, probably till 9 total 
      DOMElement* tb = 0;
      DOMElement* tbgs = 0;
      for (int PAC = 0; PAC <= 4; ++PAC  ) { // same here
        
         DOMElement* pac = 0;
        // for (int segment = 0; segment <= 11; ++segment ) {      
           std::vector< RPCMuonExtraStruct >::iterator it = m_muons.begin();
           while ( it != m_muons.end()) {
             int muSegment =  it->_mu.getLogSegment();
             //if 
             
             int muTBno = m_trigConfig->getTBNum( it->_mu.getConeCrdnts() );
             int muPACChipNo = m_trigConfig->getTowerNumOnTb(it->_mu.getConeCrdnts() );  
             int muTC =   m_trigConfig->getTCNum(it->_mu.getConeCrdnts() );
             
             if ( 
                 !( 
                        ( int(it->_level) == 0  && tbNum == muTBno &&  muTC == tcNum &&   PAC == muPACChipNo)  
                     || ( int(it->_level) == 1  && tbNum == muTBno &&  muTC == tcNum )
                     || ( int(it->_level) == 2  &&  muTC == tcNum )  
                  )
                )
  
             {
               ++it;
               continue;
             }
            // std::cout << int(it->_level) << int(it->_region) << int(it->_hsHalf)
              //         << " " << int(it->_index)
                //       << " " << it->_mu.printDebugInfo(2) << std::endl;
             

             if (tc==0) {
               tc = m_doc->createElement(X("tc"));
               currRelBx->appendChild(tc);
               tc->setAttribute(X("num"), X( IntToString(tcNum).c_str()));

               tcgs = m_doc->createElement(X("tcgs"));
               tc->appendChild(tcgs);
               
             }
             if (tb==0 && int(it->_level) <= 1) { 
               tb = m_doc->createElement(X("tb"));
               tc->appendChild(tb);
               tb->setAttribute(X("num"), X( IntToString(tbNum).c_str()));
               
               tbgs = m_doc->createElement(X("tbgs"));
               tb->appendChild(tbgs);
             }

             if (pac == 0 && int(it->_level) == 0) {
               pac =m_doc->createElement(X("pac"));
               tb->appendChild(pac);
               pac->setAttribute(X("num"), X( IntToString(muPACChipNo).c_str()));
             }
             
             DOMElement* mu = m_doc->createElement(X("mu"));
             mu->setAttribute(X("pt"), X( IntToString( int(it->_mu.getPtCode() ) ).c_str()));
             mu->setAttribute(X("qual"), X( IntToString( int(it->_mu.getQuality() ) ).c_str()));
             mu->setAttribute(X("sign"), X( IntToString( int(it->_mu.getSign() ) ).c_str()));

             if (int(it->_level) == 0 ) {
               mu->setAttribute(X("num"), X( IntToString( muSegment ).c_str()));
               pac->appendChild(mu); 
             } else {
                mu->setAttribute(X("num"), X( IntToString( int(it->_index) ).c_str()));
                mu->setAttribute(X("phi"), X( IntToString( int(it->_mu.getPhiAddr() ) ).c_str()));
                mu->setAttribute(X("eta"), X( IntToString( int(it->_mu.getEtaAddr() ) ).c_str()));
                mu->setAttribute(X("gbD"), X( IntToString( int(it->_mu.getGBData()  ) ).c_str()));
                if (int(it->_level) == 1 ) { 
                  tbgs->appendChild(mu); 
                } else if (int(it->_level) == 2 ) {
                   tcgs->appendChild(mu);
                } else {
                  throw cms::Exception("RPCMuonsGrabber") << "xx Unexpected level" << std::endl;
                }
             }

             
             it = m_muons.erase(it);             
             
           } // muons iter
        // } // segment
      } // PAC 
    } // TB
  } // TC

  

  
  for (int level=3; level<=4;++level)  {
    for (int half =0; half <= 1; ++half){
      for (int be =0; be <= 1; ++be){ // brl/endcap
    
      std::vector< RPCMuonExtraStruct >::iterator it = m_muons.begin();
      DOMElement* hs = 0;
      while ( it != m_muons.end()) {
          if ( (int(it->_level) != level) || int(it->_hsHalf)!=half  || int(it->_region)!=be )  {
            ++it;
            continue;
          }
          
          if (hs == 0) {
            if (level == 3) { 
              hs = m_doc->createElement(X("hs"));
              hs->setAttribute(X("num"), X( IntToString(half).c_str()));
            } else  if (level ==4 ) {
              hs = m_doc->createElement(X("fs"));
            } else { // shoudlnt get here
              throw cms::Exception("RPCMuonsGrabber") << "Problem writing out muons - lvl " << level << std::endl;
            }
            hs->setAttribute(X("be"), X( IntToString(be).c_str()));
            currRelBx->appendChild(hs);
          }
          
          DOMElement* mu = m_doc->createElement(X("mu"));
          hs->appendChild(mu);
          mu->setAttribute(X("num"), X( IntToString( int(it->_index) ).c_str()));
          mu->setAttribute(X("pt"), X( IntToString( int(it->_mu.getPtCode() ) ).c_str()));
          mu->setAttribute(X("qual"), X( IntToString( int(it->_mu.getQuality() ) ).c_str()));
          mu->setAttribute(X("sign"), X( IntToString( int(it->_mu.getSign() ) ).c_str()));
          mu->setAttribute(X("phi"), X( IntToString( int(it->_mu.getPhiAddr() ) ).c_str()));
          mu->setAttribute(X("eta"), X( IntToString( int(it->_mu.getEtaAddr() ) ).c_str()));
          mu->setAttribute(X("gbD"), X( IntToString( int(it->_mu.getGBData()  ) ).c_str()));
          
            //std::cout << int(it->_level) << int(it->_region) << int(it->_hsHalf) 
            //   << " " << int(it->_index) 
            //   << " " << it->_mu.printDebugInfo(2) << std::endl;

          it = m_muons.erase(it);

        } // muons iter  
      } // be iter
    } //half iteration
  } // lvl iteration

  if (m_muons.size()!=0) {
     throw cms::Exception("RPCMuonsGrabber") << " There are still some muons in muons vec" << std::endl;
  
  }
}
      


std::string MuonsGrabber::IntToString(int i){

   std::stringstream ss;
    ss << i;

     return ss.str();

}

