#ifndef LinkDataXMLWriter_h
#define LinkDataXMLWriter_h
/* This work is heavly based on KB code (L1RpcPatternXMLWriter)*/


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <FWCore/ParameterSet/interface/InputTag.h>

#include "IORawData/RPCFileReader/interface/RPCPacData.h"

#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include "boost/bind.hpp"

#include <vector>
#include <string>


XERCES_CPP_NAMESPACE_USE
/*############################################################################
#
#
#
############################################################################*/
class LinkDataXMLWriter: public edm::EDAnalyzer
{

   public:
  explicit LinkDataXMLWriter(const edm::ParameterSet&);
  virtual ~LinkDataXMLWriter();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);



   private:

  void addLinkData(int triggerCrateNum, int triggerBoardNum, 
		   int opticalLinkNum, int lbNumber, 
		   int partitionNumber,  int partitionData, 
		   int halfPart, int eod);

  void writeLinkData();  
  
  std::pair<int,int> getTCandTBNumbers(int dccInputChannelNum, int dccFactor);

  void clear();
	
  std::string m_xmlDir;
  edm::InputTag digiLabel;

  static int m_instanceCount;

  std::string IntToString(int i, int opt=0);

  int nEvents;
  int nTC, nTB;
  DOMWriter*  theSerializer;
  XMLFormatTarget *myFormTarget;
  DOMDocument* doc;
  DOMElement* rootElem;
  DOMElement* event;
  //DOMElement* oldBX;

  std::vector<std::vector<std::vector<std::vector< RPCPacData> > > > linkData;


};


#endif
