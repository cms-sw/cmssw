#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/XMLConfigWriter.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/AlgoMuon.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPattern.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternResult.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/GoldenPatternWithStat.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFConfiguration.h"
#include "L1Trigger/L1TMuonOverlapPhase1/interface/Omtf/OMTFinput.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

#include <iostream>
#include <sstream>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <bitset>

#include "xercesc/framework/StdOutFormatTarget.hpp"
#include "xercesc/framework/LocalFileFormatTarget.hpp"
#include "xercesc/parsers/XercesDOMParser.hpp"
#include "xercesc/dom/DOM.hpp"
#include "xercesc/dom/DOMException.hpp"
#include "xercesc/dom/DOMImplementation.hpp"
#include "xercesc/sax/HandlerBase.hpp"
#include "xercesc/util/XMLString.hpp"
#include "xercesc/util/PlatformUtils.hpp"
#include "xercesc/util/XercesDefs.hpp"
#include "xercesc/util/XercesVersion.hpp"
XERCES_CPP_NAMESPACE_USE

#if _XERCES_VERSION < 30100
#include "xercesc/dom/DOMWriter.hpp"
#else
#include "xercesc/dom/DOMLSSerializer.hpp"
#include "xercesc/dom/DOMLSOutput.hpp"
#endif
//

namespace {
  unsigned int eta2Bits(unsigned int eta) {
    if (eta == 73)
      return 0b100000000;
    else if (eta == 78)
      return 0b010000000;
    else if (eta == 85)
      return 0b001000000;
    else if (eta == 90)
      return 0b000100000;
    else if (eta == 94)
      return 0b000010000;
    else if (eta == 99)
      return 0b000001000;
    else if (eta == 103)
      return 0b000000100;
    else if (eta == 110)
      return 0b000000010;
    else if (eta == 75)
      return 0b110000000;
    else if (eta == 79)
      return 0b011000000;
    else if (eta == 92)
      return 0b000110000;
    else if (eta == 115)
      return 0b000000001;
    else if (eta == 121)
      return 0b000000000;
    else
      return 0b111111111;
    ;
  }
}  // namespace

//////////////////////////////////
// XMLConfigWriter
//////////////////////////////////
inline std::string _toString(XMLCh const* toTranscode) {
  std::string tmp(xercesc::XMLString::transcode(toTranscode));
  return tmp;
}

inline XMLCh* _toDOMS(std::string temp) {
  XMLCh* buff = XMLString::transcode(temp.c_str());
  return buff;
}
////////////////////////////////////
////////////////////////////////////
XMLConfigWriter::XMLConfigWriter(const OMTFConfiguration* aOMTFConfig, bool writePdfThresholds, bool writeMeanDistPhi1)
    : writePdfThresholds(writePdfThresholds), writeMeanDistPhi1(writeMeanDistPhi1) {
  XMLPlatformUtils::Initialize();

  ///Initialise XML document
  domImpl = DOMImplementationRegistry::getDOMImplementation(_toDOMS("Range"));

  myOMTFConfig = aOMTFConfig;
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////
void XMLConfigWriter::initialiseXMLDocument(const std::string& docName) {
  theDoc = domImpl->createDocument(nullptr, _toDOMS(docName), nullptr);
  theTopElement = theDoc->getDocumentElement();

  unsigned int version = myOMTFConfig->patternsVersion();
  unsigned int mask16bits = 0xFFFF;

  version &= mask16bits;

  std::ostringstream stringStr;
  stringStr.str("");
  stringStr << "0x" << std::hex << std::setfill('0') << std::setw(4) << version;
  theTopElement->setAttribute(_toDOMS("version"), _toDOMS(stringStr.str()));
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////
void XMLConfigWriter::finaliseXMLDocument(const std::string& fName) {
  XMLFormatTarget* formTarget = new LocalFileFormatTarget(fName.c_str());

#if _XERCES_VERSION < 30100
  xercesc::DOMWriter* domWriter = (dynamic_cast<DOMImplementation*>(domImpl))->createDOMWriter();
  if (domWriter->canSetFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true)) {
    domWriter->setFeature(XMLUni::fgDOMWRTFormatPrettyPrint, true);
  }
  domWriter->writeNode(formTarget, *theTopElement);

#else
  xercesc::DOMLSSerializer* theSerializer = (dynamic_cast<DOMImplementation*>(domImpl))->createLSSerializer();
  if (theSerializer->getDomConfig()->canSetParameter(XMLUni::fgDOMWRTFormatPrettyPrint, true))
    theSerializer->getDomConfig()->setParameter(XMLUni::fgDOMWRTFormatPrettyPrint, true);
  DOMLSOutput* theOutput = (dynamic_cast<DOMImplementation*>(domImpl))->createLSOutput();
  theOutput->setByteStream(formTarget);
  theSerializer->write(theTopElement, theOutput);
  theOutput->release();
  theSerializer->release();
#endif

  delete formTarget;
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////
xercesc::DOMElement* XMLConfigWriter::writeEventHeader(unsigned int eventId, unsigned int mixedEventId) {
  unsigned int eventBx = eventId * 2;

  xercesc::DOMElement* aEvent = nullptr;
  xercesc::DOMElement* aBx = nullptr;
  std::ostringstream stringStr;

  aEvent = theDoc->createElement(_toDOMS("Event"));

  stringStr.str("");
  stringStr << eventId;
  aEvent->setAttribute(_toDOMS("iEvent"), _toDOMS(stringStr.str()));

  stringStr.str("");
  stringStr << mixedEventId;
  aEvent->setAttribute(_toDOMS("iMixedEvent"), _toDOMS(stringStr.str()));

  aBx = theDoc->createElement(_toDOMS("bx"));
  stringStr.str("");
  stringStr << eventBx;
  aBx->setAttribute(_toDOMS("iBx"), _toDOMS(stringStr.str()));
  aEvent->appendChild(aBx);

  theTopElement->appendChild(aEvent);

  return aBx;
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////
xercesc::DOMElement* XMLConfigWriter::writeEventData(xercesc::DOMElement* aTopElement,
                                                     const OmtfName& board,
                                                     const OMTFinput& aInput) {
  std::ostringstream stringStr;

  xercesc::DOMElement* aProcessor = theDoc->createElement(_toDOMS("Processor"));
  aProcessor->setAttribute(_toDOMS("board"), _toDOMS(board.name()));

  unsigned int iProcessor = board.processor();
  stringStr.str("");
  stringStr << iProcessor;
  aProcessor->setAttribute(_toDOMS("iProcessor"), _toDOMS(stringStr.str()));
  stringStr.str("");
  if (board.position() == 1)
    stringStr << "+";
  stringStr << board.position();
  aProcessor->setAttribute(_toDOMS("position"), _toDOMS(stringStr.str()));

  xercesc::DOMElement *aLayer, *aHit;
  for (unsigned int iLayer = 0; iLayer < myOMTFConfig->nLayers(); ++iLayer) {
    aLayer = theDoc->createElement(_toDOMS("Layer"));
    stringStr.str("");
    stringStr << iLayer;
    aLayer->setAttribute(_toDOMS("iLayer"), _toDOMS(stringStr.str()));
    for (unsigned int iHit = 0; iHit < aInput.getMuonStubs()[iLayer].size(); ++iHit) {
      int hitPhi = aInput.getPhiHw(iLayer, iHit);
      if (hitPhi >= (int)myOMTFConfig->nPhiBins())
        continue;

      aHit = theDoc->createElement(_toDOMS("Hit"));
      stringStr.str("");
      stringStr << iHit;
      aHit->setAttribute(_toDOMS("iInput"), _toDOMS(stringStr.str()));
      stringStr.str("");
      stringStr << hitPhi;
      aHit->setAttribute(_toDOMS("iPhi"), _toDOMS(stringStr.str()));
      stringStr.str("");
      stringStr << eta2Bits(abs(aInput.getHitEta(iLayer, iHit)));
      aHit->setAttribute(_toDOMS("iEta"), _toDOMS(stringStr.str()));

      aLayer->appendChild(aHit);
    }
    if (aLayer->getChildNodes()->getLength())
      aProcessor->appendChild(aLayer);
  }

  aTopElement->appendChild(aProcessor);
  return aProcessor;
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////
void XMLConfigWriter::writeAlgoMuon(xercesc::DOMElement* aTopElement, const AlgoMuon& aCand) {
  xercesc::DOMElement* aResult = theDoc->createElement(_toDOMS("AlgoMuon"));
  std::ostringstream stringStr;
  stringStr.str("");
  stringStr << aCand.getRefHitNumber();
  aResult->setAttribute(_toDOMS("iRefHit"), _toDOMS(stringStr.str()));
  stringStr.str("");
  stringStr << aCand.getPt();
  aResult->setAttribute(_toDOMS("ptCode"), _toDOMS(stringStr.str()));
  stringStr.str("");
  stringStr << aCand.getPhi();
  aResult->setAttribute(_toDOMS("phiCode"), _toDOMS(stringStr.str()));
  stringStr.str("");
  stringStr << eta2Bits(abs(aCand.getEtaHw()));
  aResult->setAttribute(_toDOMS("etaCode"), _toDOMS(stringStr.str()));
  stringStr.str("");
  stringStr << aCand.getCharge();
  aResult->setAttribute(_toDOMS("charge"), _toDOMS(stringStr.str()));
  stringStr.str("");
  stringStr << aCand.getQ();
  aResult->setAttribute(_toDOMS("nHits"), _toDOMS(stringStr.str()));
  stringStr.str("");
  stringStr << aCand.getDisc();
  aResult->setAttribute(_toDOMS("disc"), _toDOMS(stringStr.str()));
  stringStr.str("");
  stringStr << aCand.getRefLayer();
  aResult->setAttribute(_toDOMS("iRefLayer"), _toDOMS(stringStr.str()));
  stringStr.str("");
  stringStr << std::bitset<18>(aCand.getFiredLayerBits());
  aResult->setAttribute(_toDOMS("layers"), _toDOMS(stringStr.str()));
  stringStr.str("");
  stringStr << aCand.getPhiRHit();
  aResult->setAttribute(_toDOMS("phiRHit"), _toDOMS(stringStr.str()));
  stringStr.str("");
  stringStr << aCand.getHwPatternNumber();
  aResult->setAttribute(_toDOMS("patNum"), _toDOMS(stringStr.str()));

  aTopElement->appendChild(aResult);
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////
void XMLConfigWriter::writeCandMuon(xercesc::DOMElement* aTopElement, const l1t::RegionalMuonCand& aCand) {
  xercesc::DOMElement* aResult = theDoc->createElement(_toDOMS("CandMuon"));
  std::ostringstream stringStr;
  stringStr.str("");
  stringStr << aCand.hwPt();
  aResult->setAttribute(_toDOMS("hwPt"), _toDOMS(stringStr.str()));
  stringStr.str("");
  stringStr << aCand.hwPhi();
  aResult->setAttribute(_toDOMS("hwPhi"), _toDOMS(stringStr.str()));
  stringStr.str("");
  stringStr << aCand.hwEta();
  aResult->setAttribute(_toDOMS("hwEta"), _toDOMS(stringStr.str()));
  stringStr.str("");
  stringStr << aCand.hwSign();
  aResult->setAttribute(_toDOMS("hwSign"), _toDOMS(stringStr.str()));
  stringStr.str("");
  stringStr << aCand.hwSignValid();
  aResult->setAttribute(_toDOMS("hwSignValid"), _toDOMS(stringStr.str()));
  stringStr.str("");
  stringStr << aCand.hwQual();
  aResult->setAttribute(_toDOMS("hwQual"), _toDOMS(stringStr.str()));
  stringStr.str("");
  std::map<int, int> hwAddrMap = aCand.trackAddress();
  stringStr << std::bitset<29>(hwAddrMap[0]);
  aResult->setAttribute(_toDOMS("hwTrackAddress"), _toDOMS(stringStr.str()));
  stringStr.str("");
  stringStr << aCand.link();
  aResult->setAttribute(_toDOMS("link"), _toDOMS(stringStr.str()));
  stringStr.str("");
  stringStr << aCand.processor();
  aResult->setAttribute(_toDOMS("processor"), _toDOMS(stringStr.str()));
  stringStr.str("");
  if (aCand.trackFinderType() == l1t::omtf_neg)
    stringStr << "OMTF_NEG";
  else if (aCand.trackFinderType() == l1t::omtf_pos)
    stringStr << "OMTF_POS";
  else
    stringStr << aCand.trackFinderType();
  aResult->setAttribute(_toDOMS("trackFinderType"), _toDOMS(stringStr.str()));
  aTopElement->appendChild(aResult);
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////
//////////////////////////////////////////////////
//////////////////////////////////////////////////
void XMLConfigWriter::writeResultsData(xercesc::DOMElement* aTopElement,
                                       unsigned int iRegion,
                                       const Key& aKey,
                                       const GoldenPatternResult& aResult) {
  std::ostringstream stringStr;
  ///Write GP key parameters
  xercesc::DOMElement* aGP = theDoc->createElement(_toDOMS("GP"));
  stringStr.str("");
  stringStr << aKey.thePt;
  aGP->setAttribute(_toDOMS("iPt"), _toDOMS(stringStr.str()));
  stringStr.str("");
  stringStr << aKey.theEtaCode;
  aGP->setAttribute(_toDOMS("iEta"), _toDOMS(stringStr.str()));
  stringStr.str("");
  stringStr << "0";
  aGP->setAttribute(_toDOMS("iPhi"), _toDOMS(stringStr.str()));
  stringStr.str("");
  stringStr << aKey.theCharge;
  aGP->setAttribute(_toDOMS("iCharge"), _toDOMS(stringStr.str()));
  /////////////////
  ///Write results details for this GP
  {
    xercesc::DOMElement* aRefLayer = theDoc->createElement(_toDOMS("Result"));
    stringStr.str("");
    stringStr << aResult.getRefLayer();
    aRefLayer->setAttribute(_toDOMS("iRefLayer"), _toDOMS(stringStr.str()));
    stringStr.str("");
    stringStr << iRegion;
    aRefLayer->setAttribute(_toDOMS("iRegion"), _toDOMS(stringStr.str()));
    stringStr.str("");
    stringStr << myOMTFConfig->getRefToLogicNumber()[aResult.getRefLayer()];
    aRefLayer->setAttribute(_toDOMS("iLogicLayer"), _toDOMS(stringStr.str()));
    for (unsigned int iLogicLayer = 0; iLogicLayer < myOMTFConfig->nLayers(); ++iLogicLayer) {
      xercesc::DOMElement* aLayer = theDoc->createElement(_toDOMS("Layer"));
      stringStr.str("");
      stringStr << iLogicLayer;
      aLayer->setAttribute(_toDOMS("iLayer"), _toDOMS(stringStr.str()));
      stringStr.str("");
      stringStr << aResult.getStubResults()[iLogicLayer].getPdfVal();
      aLayer->setAttribute(_toDOMS("value"), _toDOMS(stringStr.str()));
      if (aResult.getStubResults()[iLogicLayer].getPdfVal())
        aRefLayer->appendChild(aLayer);
    }
    if (aRefLayer->getChildNodes()->getLength())
      aGP->appendChild(aRefLayer);
  }
  if (aGP->getChildNodes()->getLength())
    aTopElement->appendChild(aGP);
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////
void XMLConfigWriter::writeGPData(GoldenPattern& aGP) {
  std::ostringstream stringStr;
  xercesc::DOMElement *aLayer = nullptr, *aRefLayer = nullptr, *aPdf = nullptr;

  xercesc::DOMElement* aGPElement = theDoc->createElement(_toDOMS("GP"));
  stringStr.str("");
  stringStr << aGP.key().thePt;
  aGPElement->setAttribute(_toDOMS("iPt"), _toDOMS(stringStr.str()));
  stringStr.str("");
  //stringStr<<aGP.key().theEtaCode;
  stringStr << "0";  //No eta code at the moment
  aGPElement->setAttribute(_toDOMS("iEta"), _toDOMS(stringStr.str()));
  stringStr.str("");
  stringStr << 0;  //No phi code is assigned to GP for the moment.
  aGPElement->setAttribute(_toDOMS("iPhi"), _toDOMS(stringStr.str()));
  stringStr.str("");
  stringStr << aGP.key().theCharge;
  aGPElement->setAttribute(_toDOMS("iCharge"), _toDOMS(stringStr.str()));

  for (unsigned int iLayer = 0; iLayer < myOMTFConfig->nLayers(); ++iLayer) {
    int nOfPhis = 0;
    /////////////////////////////////////
    aLayer = theDoc->createElement(_toDOMS("Layer"));
    stringStr.str("");
    stringStr << iLayer;
    //////////////////////////////////
    aLayer->setAttribute(_toDOMS("iLayer"), _toDOMS(stringStr.str()));
    stringStr.str("");
    stringStr << nOfPhis;
    aLayer->setAttribute(_toDOMS("nOfPhis"), _toDOMS(stringStr.str()));
    for (unsigned int iRefLayer = 0; iRefLayer < myOMTFConfig->nRefLayers(); ++iRefLayer) {
      aRefLayer = theDoc->createElement(_toDOMS("RefLayer"));
      int meanDistPhi0 = aGP.getMeanDistPhi()[iLayer][iRefLayer][0];
      stringStr.str("");
      stringStr << meanDistPhi0;
      aRefLayer->setAttribute(_toDOMS("meanDistPhi0"), _toDOMS(stringStr.str()));

      int meanDistPhi1 = aGP.getMeanDistPhi()[iLayer][iRefLayer][1];
      stringStr.str("");
      stringStr << meanDistPhi1;
      aRefLayer->setAttribute(_toDOMS("meanDistPhi1"), _toDOMS(stringStr.str()));

      int selDistPhi = 0;
      stringStr.str("");
      stringStr << selDistPhi;
      aRefLayer->setAttribute(_toDOMS("selDistPhi"), _toDOMS(stringStr.str()));

      int selDistPhiShift =
          aGP.getDistPhiBitShift(iLayer, iRefLayer);  //TODO check if Wojtek expects it here or on the distMsbPhiShift
      stringStr.str("");
      stringStr << selDistPhiShift;
      aRefLayer->setAttribute(_toDOMS("selDistPhiShift"), _toDOMS(stringStr.str()));

      int distMsbPhiShift = 0;
      stringStr.str("");
      stringStr << distMsbPhiShift;
      aRefLayer->setAttribute(_toDOMS("distMsbPhiShift"), _toDOMS(stringStr.str()));
      aLayer->appendChild(aRefLayer);
    }
    for (unsigned int iRefLayer = 0; iRefLayer < myOMTFConfig->nRefLayers(); ++iRefLayer) {
      for (unsigned int iPdf = 0; iPdf < exp2(myOMTFConfig->nPdfAddrBits()); ++iPdf) {
        aPdf = theDoc->createElement(_toDOMS("PDF"));
        stringStr.str("");
        stringStr << aGP.pdfValue(iLayer, iRefLayer, iPdf);
        aPdf->setAttribute(_toDOMS("value"), _toDOMS(stringStr.str()));
        aLayer->appendChild(aPdf);
      }
    }
    aGPElement->appendChild(aLayer);
  }
  theTopElement->appendChild(aGPElement);
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////
void XMLConfigWriter::writeGPData(GoldenPattern* aGP1, GoldenPattern* aGP2, GoldenPattern* aGP3, GoldenPattern* aGP4) {
  std::ostringstream stringStr;
  auto setAttributeInt = [&](xercesc::DOMElement* domElement, std::string name, int value) -> void {
    stringStr << value;
    domElement->setAttribute(_toDOMS(name), _toDOMS(stringStr.str()));
    stringStr.str("");
  };

  auto setAttributeFloat = [&](xercesc::DOMElement* domElement, std::string name, float value) -> void {
    stringStr << value;
    domElement->setAttribute(_toDOMS(name), _toDOMS(stringStr.str()));
    stringStr.str("");
  };

  //xercesc::DOMElement *aLayer=nullptr, *aRefLayer=nullptr, *aPdf=nullptr;

  xercesc::DOMElement* aGPElement = theDoc->createElement(_toDOMS("GP"));

  setAttributeInt(aGPElement, "iPt1", aGP1->key().thePt);
  setAttributeInt(aGPElement, "iPt2", aGP2->key().thePt);
  setAttributeInt(aGPElement, "iPt3", aGP3->key().thePt);
  setAttributeInt(aGPElement, "iPt4", aGP4->key().thePt);

  if (writePdfThresholds) {
    if (dynamic_cast<const GoldenPatternWithThresh*>(aGP1) != nullptr) {
      for (unsigned int iRefLayer = 0; iRefLayer < myOMTFConfig->nRefLayers(); ++iRefLayer) {
        //cout<<__FUNCTION__<<":"<<__LINE__<<std::endl;
        xercesc::DOMElement* aRefLayerThresh = theDoc->createElement(_toDOMS("RefLayerThresh"));
        setAttributeFloat(
            aRefLayerThresh, "tresh1", dynamic_cast<const GoldenPatternWithThresh*>(aGP1)->getThreshold(iRefLayer));
        setAttributeFloat(
            aRefLayerThresh, "tresh2", dynamic_cast<const GoldenPatternWithThresh*>(aGP2)->getThreshold(iRefLayer));
        setAttributeFloat(
            aRefLayerThresh, "tresh3", dynamic_cast<const GoldenPatternWithThresh*>(aGP3)->getThreshold(iRefLayer));
        setAttributeFloat(
            aRefLayerThresh, "tresh4", dynamic_cast<const GoldenPatternWithThresh*>(aGP4)->getThreshold(iRefLayer));

        aGPElement->appendChild(aRefLayerThresh);
      }
    }
  }

  setAttributeInt(aGPElement, "iEta", 0);  //aGP1.key().theEtaCode; //No eta code at the moment

  setAttributeInt(aGPElement, "iPhi", 0);  //No phi code is assigned to GP for the moment.

  setAttributeInt(aGPElement, "iCharge", aGP1->key().theCharge);

  for (unsigned int iLayer = 0; iLayer < myOMTFConfig->nLayers(); ++iLayer) {
    int nOfPhis = 0;
    /////////////////////////////////////
    xercesc::DOMElement* aLayer = theDoc->createElement(_toDOMS("Layer"));

    setAttributeInt(aLayer, "iLayer", iLayer);
    setAttributeInt(aLayer, "nOfPhis", nOfPhis);

    for (unsigned int iRefLayer = 0; iRefLayer < myOMTFConfig->nRefLayers(); ++iRefLayer) {
      xercesc::DOMElement* aRefLayer = theDoc->createElement(_toDOMS("RefLayer"));

      if (writeMeanDistPhi1) {
        int meanDistPhi0 = aGP1->getMeanDistPhi()[iLayer][iRefLayer][0];
        setAttributeInt(aRefLayer, "meanDistPhi0", meanDistPhi0);

        int meanDistPhi1 = aGP1->getMeanDistPhi()[iLayer][iRefLayer][1];
        setAttributeInt(aRefLayer, "meanDistPhi1", meanDistPhi1);
      } else {
        int meanDistPhi = aGP1->getMeanDistPhi()[iLayer][iRefLayer][0];
        setAttributeInt(aRefLayer, "meanDistPhi", meanDistPhi);
      }

      int selDistPhi = 0;
      setAttributeInt(aRefLayer, "selDistPhi", selDistPhi);

      int selDistPhiShift = aGP1->getDistPhiBitShift(
          iLayer, iRefLayer);  //TODO check if Wojtek expects it here or on the distMsbPhiShift;
      setAttributeInt(aRefLayer, "selDistPhiShift", selDistPhiShift);

      int distMsbPhiShift = 0;
      setAttributeInt(aRefLayer, "distMsbPhiShift", distMsbPhiShift);

      aLayer->appendChild(aRefLayer);
    }
    for (unsigned int iRefLayer = 0; iRefLayer < myOMTFConfig->nRefLayers(); ++iRefLayer) {
      for (unsigned int iPdf = 0; iPdf < exp2(myOMTFConfig->nPdfAddrBits()); ++iPdf) {
        xercesc::DOMElement* aPdf = theDoc->createElement(_toDOMS("PDF"));

        setAttributeFloat(aPdf, "value1", aGP1->pdfValue(iLayer, iRefLayer, iPdf));
        setAttributeFloat(aPdf, "value2", aGP2->pdfValue(iLayer, iRefLayer, iPdf));
        setAttributeFloat(aPdf, "value3", aGP3->pdfValue(iLayer, iRefLayer, iPdf));
        setAttributeFloat(aPdf, "value4", aGP4->pdfValue(iLayer, iRefLayer, iPdf));

        aLayer->appendChild(aPdf);
      }
    }
    aGPElement->appendChild(aLayer);
  }
  theTopElement->appendChild(aGPElement);
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////
template <class GoldenPatternType>
void XMLConfigWriter::writeGPs(const std::vector<std::shared_ptr<GoldenPatternType> >& goldenPats, std::string fName) {
  initialiseXMLDocument("OMTF");
  GoldenPattern* dummy = new GoldenPatternWithThresh(Key(0, 0, 0), myOMTFConfig);

  OMTFConfiguration::vector2D mergedPartters = myOMTFConfig->getPatternGroups(goldenPats);
  for (unsigned int iGroup = 0; iGroup < mergedPartters.size(); iGroup++) {
    std::vector<GoldenPattern*> gps(4, dummy);
    for (unsigned int i = 0; i < mergedPartters[iGroup].size(); i++) {
      GoldenPattern* gp = dynamic_cast<GoldenPattern*>(goldenPats.at(mergedPartters[iGroup][i]).get());
      if (!gp) {
        throw cms::Exception("OMTF::XMLConfigWriter::writeGPs: the gps are not GoldenPatterns ");
      }
      /*cout<<gp->key()<<endl;;
      for(unsigned int iLayer = 0; iLayer<myOMTFConfig->nLayers(); ++iLayer) {
        for(unsigned int iRefLayer=0; iRefLayer<myOMTFConfig->nRefLayers(); ++iRefLayer) {
          if(gp->getPdf()[iLayer][iRefLayer][0] != 0) {
            cout<<"iLayer "<<iLayer<<" iRefLayer "<<iRefLayer<<" pdf[0] "<<gp->getPdf()[iLayer][iRefLayer][0]<<"!!!!!!!!!!!!!!!!!!!!\n";
          }
        }
      }*/
      gps[i] = gp;
    }
    writeGPData(gps[0], gps[1], gps[2], gps[3]);
  }
  finaliseXMLDocument(fName);
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////
void XMLConfigWriter::writeConnectionsData(
    const std::vector<std::vector<OMTFConfiguration::vector2D> >& measurements4D) {
  std::ostringstream stringStr;

  for (unsigned int iProcessor = 0; iProcessor < 6; ++iProcessor) {
    xercesc::DOMElement* aProcessorElement = theDoc->createElement(_toDOMS("Processor"));
    stringStr.str("");
    stringStr << iProcessor;
    aProcessorElement->setAttribute(_toDOMS("iProcessor"), _toDOMS(stringStr.str()));
    for (unsigned int iRefLayer = 0; iRefLayer < myOMTFConfig->nRefLayers(); ++iRefLayer) {
      xercesc::DOMElement* aRefLayerElement = theDoc->createElement(_toDOMS("RefLayer"));
      stringStr.str("");
      stringStr << iRefLayer;
      aRefLayerElement->setAttribute(_toDOMS("iRefLayer"), _toDOMS(stringStr.str()));
      stringStr.str("");
      stringStr << myOMTFConfig->getProcessorPhiVsRefLayer()[iProcessor][iRefLayer];
      aRefLayerElement->setAttribute(_toDOMS("iGlobalPhiStart"), _toDOMS(stringStr.str()));
      aProcessorElement->appendChild(aRefLayerElement);
    }
    unsigned int iRefHit = 0;
    ///////
    for (unsigned int iRefLayer = 0; iRefLayer < myOMTFConfig->nRefLayers(); ++iRefLayer) {
      for (unsigned int iRegion = 0; iRegion < 6; ++iRegion) {
        unsigned int maxHitCount = 0;
        for (unsigned int iInput = 0; iInput < 14; ++iInput) {
          if ((int)maxHitCount < myOMTFConfig->getMeasurements4Dref()[iProcessor][iRegion][iRefLayer][iInput])
            maxHitCount = myOMTFConfig->getMeasurements4Dref()[iProcessor][iRegion][iRefLayer][iInput];
        }
        for (unsigned int iInput = 0; iInput < 14; ++iInput) {
          unsigned int hitCount = myOMTFConfig->getMeasurements4Dref()[iProcessor][iRegion][iRefLayer][iInput];
          if (hitCount < maxHitCount * 0.1)
            continue;
          xercesc::DOMElement* aRefHitElement = theDoc->createElement(_toDOMS("RefHit"));
          stringStr.str("");
          stringStr << iRefHit;
          aRefHitElement->setAttribute(_toDOMS("iRefHit"), _toDOMS(stringStr.str()));
          stringStr.str("");
          stringStr << iRefLayer;
          aRefHitElement->setAttribute(_toDOMS("iRefLayer"), _toDOMS(stringStr.str()));

          stringStr.str("");
          stringStr << iRegion;
          aRefHitElement->setAttribute(_toDOMS("iRegion"), _toDOMS(stringStr.str()));

          stringStr.str("");
          stringStr << iInput;
          aRefHitElement->setAttribute(_toDOMS("iInput"), _toDOMS(stringStr.str()));
          unsigned int logicRegionSize = 10 / 360.0 * myOMTFConfig->nPhiBins();
          int lowScaleEnd = std::pow(2, myOMTFConfig->nPhiBits() - 1);
          ///iPhiMin and iPhiMax are expressed in n bit scale -2**n, +2**2-1 used in each processor
          int iPhiMin = myOMTFConfig->getProcessorPhiVsRefLayer()[iProcessor][iRefLayer] -
                        myOMTFConfig->globalPhiStart(iProcessor) - lowScaleEnd;
          int iPhiMax = iPhiMin + logicRegionSize - 1;

          iPhiMin += iRegion * logicRegionSize;
          iPhiMax += iRegion * logicRegionSize;

          stringStr.str("");
          stringStr << iPhiMin;
          aRefHitElement->setAttribute(_toDOMS("iPhiMin"), _toDOMS(stringStr.str()));

          stringStr.str("");
          stringStr << iPhiMax;
          aRefHitElement->setAttribute(_toDOMS("iPhiMax"), _toDOMS(stringStr.str()));
          if (iRefHit < myOMTFConfig->nRefHits())
            aProcessorElement->appendChild(aRefHitElement);
          ++iRefHit;
        }
        for (; iRegion == 5 && iRefLayer == 7 && iRefHit < myOMTFConfig->nRefHits(); ++iRefHit) {
          xercesc::DOMElement* aRefHitElement = theDoc->createElement(_toDOMS("RefHit"));
          stringStr.str("");
          stringStr << iRefHit;
          aRefHitElement->setAttribute(_toDOMS("iRefHit"), _toDOMS(stringStr.str()));
          stringStr.str("");
          stringStr << 0;
          aRefHitElement->setAttribute(_toDOMS("iRefLayer"), _toDOMS(stringStr.str()));

          stringStr.str("");
          stringStr << 0;
          aRefHitElement->setAttribute(_toDOMS("iRegion"), _toDOMS(stringStr.str()));

          stringStr.str("");
          stringStr << 0;
          aRefHitElement->setAttribute(_toDOMS("iInput"), _toDOMS(stringStr.str()));

          int iPhiMin = 0;
          int iPhiMax = 1;

          stringStr.str("");
          stringStr << iPhiMin;
          aRefHitElement->setAttribute(_toDOMS("iPhiMin"), _toDOMS(stringStr.str()));

          stringStr.str("");
          stringStr << iPhiMax;
          aRefHitElement->setAttribute(_toDOMS("iPhiMax"), _toDOMS(stringStr.str()));

          aProcessorElement->appendChild(aRefHitElement);
        }
      }
    }
    ////
    for (unsigned int iRegion = 0; iRegion < 6; ++iRegion) {
      xercesc::DOMElement* aRegionElement = theDoc->createElement(_toDOMS("LogicRegion"));
      stringStr.str("");
      stringStr << iRegion;
      aRegionElement->setAttribute(_toDOMS("iRegion"), _toDOMS(stringStr.str()));

      for (unsigned int iLogicLayer = 0; iLogicLayer < myOMTFConfig->nLayers(); ++iLogicLayer) {
        xercesc::DOMElement* aLayerElement = theDoc->createElement(_toDOMS("Layer"));
        stringStr.str("");
        stringStr << iLogicLayer;
        ////////////////////////////////////////////////
        aLayerElement->setAttribute(_toDOMS("iLayer"), _toDOMS(stringStr.str()));
        const OMTFConfiguration::vector1D& myCounts =
            myOMTFConfig->getMeasurements4D()[iProcessor][iRegion][iLogicLayer];
        unsigned int maxInput = findMaxInput(myCounts);
        unsigned int begin = 0, end = 0;
        if ((int)maxInput - 2 >= 0)
          begin = maxInput - 2;
        else
          begin = maxInput;
        end = maxInput + 3;
        stringStr.str("");
        stringStr << begin;
        aLayerElement->setAttribute(_toDOMS("iFirstInput"), _toDOMS(stringStr.str()));
        stringStr.str("");
        stringStr << end - begin + 1;
        aLayerElement->setAttribute(_toDOMS("nInputs"), _toDOMS(stringStr.str()));
        aRegionElement->appendChild(aLayerElement);
      }
      aProcessorElement->appendChild(aRegionElement);
    }
    theTopElement->appendChild(aProcessorElement);
  }
}
//////////////////////////////////////////////////
//////////////////////////////////////////////////
unsigned int XMLConfigWriter::findMaxInput(const OMTFConfiguration::vector1D& myCounts) {
  unsigned int max = 0;
  unsigned int maxInput = 0;
  for (unsigned int iInput = 0; iInput < 14; ++iInput) {
    if (myCounts[iInput] > (int)max) {
      max = myCounts[iInput];
      maxInput = iInput;
    }
  }
  return maxInput;
}
/////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////
template void XMLConfigWriter::writeGPs(const std::vector<std::shared_ptr<GoldenPattern> >& goldenPats,
                                        std::string fName);

template void XMLConfigWriter::writeGPs(const std::vector<std::shared_ptr<GoldenPatternWithStat> >& goldenPats,
                                        std::string fName);


template void XMLConfigWriter::writeGPs(const std::vector<std::shared_ptr<GoldenPatternWithThresh> >& goldenPats,
                                        std::string fName);
