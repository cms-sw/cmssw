// -*- C++ -*-
//
// Package:     MuonAlignment
// Class  :     MuonAlignmentInputXML
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Jim Pivarski
//         Created:  Mon Mar 10 16:37:40 CDT 2008
//$Id: MuonAlignmentInputXML.cc,v 1.16 2011/03/22 09:49:50 innocent Exp $
//

// system include files
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Concurrency/interface/Xerces.h"

// Xerces include files
#include "xercesc/parsers/XercesDOMParser.hpp"
#include "xercesc/dom/DOM.hpp"
#include "xercesc/sax/HandlerBase.hpp"
#include "xercesc/util/XMLString.hpp"
#include "xercesc/util/PlatformUtils.hpp"
#include "xercesc/util/XercesDefs.hpp"
XERCES_CPP_NAMESPACE_USE

// user include files
#include "Alignment/MuonAlignment/interface/MuonAlignmentInputXML.h"
#include "Alignment/CommonAlignment/interface/StructureType.h"
#include "DataFormats/MuonDetId/interface/CSCDetId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
MuonAlignmentInputXML::MuonAlignmentInputXML(std::string fileName)
   : m_fileName(fileName)
{
   str_operation = XMLString::transcode("operation");
   str_collection = XMLString::transcode("collection");
   str_name = XMLString::transcode("name");
   str_DTBarrel = XMLString::transcode("DTBarrel");
   str_DTWheel = XMLString::transcode("DTWheel");
   str_DTStation = XMLString::transcode("DTStation");
   str_DTChamber = XMLString::transcode("DTChamber");
   str_DTSuperLayer = XMLString::transcode("DTSuperLayer");
   str_DTLayer = XMLString::transcode("DTLayer");
   str_CSCEndcap = XMLString::transcode("CSCEndcap");
   str_CSCStation = XMLString::transcode("CSCStation");
   str_CSCRing = XMLString::transcode("CSCRing");
   str_CSCChamber = XMLString::transcode("CSCChamber");
   str_CSCLayer = XMLString::transcode("CSCLayer");
   str_setposition = XMLString::transcode("setposition");
   str_setape = XMLString::transcode("setape");
   str_setsurveyerr = XMLString::transcode("setsurveyerr");
   str_moveglobal = XMLString::transcode("moveglobal");
   str_movelocal = XMLString::transcode("movelocal");
   str_rotatelocal = XMLString::transcode("rotatelocal");
   str_rotatebeamline = XMLString::transcode("rotatebeamline");
   str_rotateglobalaxis = XMLString::transcode("rotateglobalaxis");
   str_relativeto = XMLString::transcode("relativeto");
   str_rawId = XMLString::transcode("rawId");
   str_wheel = XMLString::transcode("wheel");
   str_station = XMLString::transcode("station");
   str_sector = XMLString::transcode("sector");
   str_superlayer = XMLString::transcode("superlayer");
   str_layer = XMLString::transcode("layer");
   str_endcap = XMLString::transcode("endcap");
   str_ring = XMLString::transcode("ring");
   str_chamber = XMLString::transcode("chamber");
   str_axisx = XMLString::transcode("axisx");
   str_axisy = XMLString::transcode("axisy");
   str_axisz = XMLString::transcode("axisz");
   str_angle = XMLString::transcode("angle");
   str_x = XMLString::transcode("x");
   str_y = XMLString::transcode("y");
   str_z = XMLString::transcode("z");
   str_phix = XMLString::transcode("phix");
   str_phiy = XMLString::transcode("phiy");
   str_phiz = XMLString::transcode("phiz");
   str_alpha = XMLString::transcode("alpha");
   str_beta = XMLString::transcode("beta");
   str_gamma = XMLString::transcode("gamma");
   str_rphi = XMLString::transcode("rphi");
   str_phi = XMLString::transcode("phi");
   str_xx = XMLString::transcode("xx");
   str_xy = XMLString::transcode("xy");
   str_xz = XMLString::transcode("xz");
   str_xa = XMLString::transcode("xa");
   str_xb = XMLString::transcode("xb");
   str_xc = XMLString::transcode("xc");
   str_yy = XMLString::transcode("yy");
   str_yz = XMLString::transcode("yz");
   str_ya = XMLString::transcode("ya");
   str_yb = XMLString::transcode("yb");
   str_yc = XMLString::transcode("yc");
   str_zz = XMLString::transcode("zz");
   str_za = XMLString::transcode("za");
   str_zb = XMLString::transcode("zb");
   str_zc = XMLString::transcode("zc");
   str_aa = XMLString::transcode("aa");
   str_ab = XMLString::transcode("ab");
   str_ac = XMLString::transcode("ac");
   str_bb = XMLString::transcode("bb");
   str_bc = XMLString::transcode("bc");
   str_cc = XMLString::transcode("cc");
   str_none = XMLString::transcode("none");
   str_ideal = XMLString::transcode("ideal");
   str_container = XMLString::transcode("container");
}

// MuonAlignmentInputXML::MuonAlignmentInputXML(const MuonAlignmentInputXML& rhs)
// {
//    // do actual copying here;
// }

MuonAlignmentInputXML::~MuonAlignmentInputXML() {
   XMLString::release(&str_operation);
   XMLString::release(&str_collection);
   XMLString::release(&str_name);
   XMLString::release(&str_DTBarrel);
   XMLString::release(&str_DTWheel);
   XMLString::release(&str_DTStation);
   XMLString::release(&str_DTChamber);
   XMLString::release(&str_DTSuperLayer);
   XMLString::release(&str_DTLayer);
   XMLString::release(&str_CSCEndcap);
   XMLString::release(&str_CSCStation);
   XMLString::release(&str_CSCRing);
   XMLString::release(&str_CSCChamber);
   XMLString::release(&str_CSCLayer);
   XMLString::release(&str_setposition);
   XMLString::release(&str_setape);
   XMLString::release(&str_setsurveyerr);
   XMLString::release(&str_moveglobal);
   XMLString::release(&str_movelocal);
   XMLString::release(&str_rotatelocal);
   XMLString::release(&str_rotatebeamline);
   XMLString::release(&str_rotateglobalaxis);
   XMLString::release(&str_relativeto);
   XMLString::release(&str_rawId);
   XMLString::release(&str_wheel);
   XMLString::release(&str_station);
   XMLString::release(&str_sector);
   XMLString::release(&str_superlayer);
   XMLString::release(&str_layer);
   XMLString::release(&str_endcap);
   XMLString::release(&str_ring);
   XMLString::release(&str_chamber);
   XMLString::release(&str_axisx);
   XMLString::release(&str_axisy);
   XMLString::release(&str_axisz);
   XMLString::release(&str_angle);
   XMLString::release(&str_x);
   XMLString::release(&str_y);
   XMLString::release(&str_z);
   XMLString::release(&str_phix);
   XMLString::release(&str_phiy);
   XMLString::release(&str_phiz);
   XMLString::release(&str_alpha);
   XMLString::release(&str_beta);
   XMLString::release(&str_gamma);
   XMLString::release(&str_rphi);
   XMLString::release(&str_phi);
   XMLString::release(&str_xx);
   XMLString::release(&str_xy);
   XMLString::release(&str_xz);
   XMLString::release(&str_xa);
   XMLString::release(&str_xb);
   XMLString::release(&str_xc);
   XMLString::release(&str_yy);
   XMLString::release(&str_yz);
   XMLString::release(&str_ya);
   XMLString::release(&str_yb);
   XMLString::release(&str_yc);
   XMLString::release(&str_zz);
   XMLString::release(&str_za);
   XMLString::release(&str_zb);
   XMLString::release(&str_zc);
   XMLString::release(&str_aa);
   XMLString::release(&str_ab);
   XMLString::release(&str_ac);
   XMLString::release(&str_bb);
   XMLString::release(&str_bc);
   XMLString::release(&str_cc);
   XMLString::release(&str_none);
   XMLString::release(&str_ideal);
   XMLString::release(&str_container);
}

//
// assignment operators
//
// const MuonAlignmentInputXML& MuonAlignmentInputXML::operator=(const MuonAlignmentInputXML& rhs)
// {
//   //An exception safe implementation is
//   MuonAlignmentInputXML temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//

void MuonAlignmentInputXML::recursiveGetId(std::map<unsigned int, Alignable*> &alignableNavigator, const align::Alignables &alignables) const {
   for (align::Alignables::const_iterator ali = alignables.begin();  ali != alignables.end();  ++ali) {
      if ((*ali)->alignableObjectId() == align::AlignableDetUnit  ||  (*ali)->alignableObjectId() == align::AlignableDet  ||
	  (*ali)->alignableObjectId() == align::AlignableDTChamber  ||  (*ali)->alignableObjectId() == align::AlignableDTSuperLayer  ||  (*ali)->alignableObjectId() == align::AlignableDTLayer  ||
	  (*ali)->alignableObjectId() == align::AlignableCSCChamber  ||  (*ali)->alignableObjectId() == align::AlignableCSCLayer) {
	 alignableNavigator[(*ali)->geomDetId().rawId()] = *ali;
      }
      recursiveGetId(alignableNavigator, (*ali)->components());
   }
}

void MuonAlignmentInputXML::fillAliToIdeal(std::map<Alignable*, Alignable*> &alitoideal, const align::Alignables& alignables, const align::Alignables& ideals) const {
   align::Alignables::const_iterator alignable = alignables.begin();
   align::Alignables::const_iterator ideal = ideals.begin();

   while (alignable != alignables.end()  &&  ideal != ideals.end()) {
      alitoideal[*alignable] = *ideal;

      fillAliToIdeal(alitoideal, (*alignable)->components(), (*ideal)->components());

      ++alignable;
      ++ideal;
   }

   if (alignable != alignables.end()  ||  ideal != ideals.end()) {
      throw cms::Exception("Alignment") << "alignable and ideal-alignable trees are out of sync (this should never happen)";
   }
}

AlignableMuon *MuonAlignmentInputXML::newAlignableMuon(const edm::EventSetup& iSetup) const {
   boost::shared_ptr<DTGeometry> dtGeometry = idealDTGeometry(iSetup);
   boost::shared_ptr<CSCGeometry> cscGeometry = idealCSCGeometry(iSetup);

   AlignableMuon *alignableMuon = new AlignableMuon(&(*dtGeometry), &(*cscGeometry));
   std::map<unsigned int, Alignable*> alignableNavigator;  // real AlignableNavigators don't have const methods
   recursiveGetId(alignableNavigator, alignableMuon->DTBarrel());
   recursiveGetId(alignableNavigator, alignableMuon->CSCEndcaps());

   AlignableMuon *ideal_alignableMuon = new AlignableMuon(&(*dtGeometry), &(*cscGeometry));
   std::map<unsigned int, Alignable*> ideal_alignableNavigator;  // real AlignableNavigators don't have const methods
   recursiveGetId(ideal_alignableNavigator, ideal_alignableMuon->DTBarrel());
   recursiveGetId(ideal_alignableNavigator, ideal_alignableMuon->CSCEndcaps());

   try {
      cms::concurrency::xercesInitialize();
   }
   catch (const XMLException &toCatch) {
      throw cms::Exception("XMLException") << "Xerces XML parser threw an exception on initialization." << std::endl;
   }

   XercesDOMParser *parser = new XercesDOMParser();
   parser->setValidationScheme(XercesDOMParser::Val_Always);

   ErrorHandler *errHandler = (ErrorHandler*)(new HandlerBase());
   parser->setErrorHandler(errHandler);

   try {
      parser->parse(m_fileName.c_str());
   }
   catch (const XMLException &toCatch) {
      char *message = XMLString::transcode(toCatch.getMessage());
      throw cms::Exception("XMLException") << "Xerces XML parser threw this exception: " << message << std::endl;
   }
   catch (const DOMException &toCatch) {
      char *message = XMLString::transcode(toCatch.msg);
      throw cms::Exception("XMLException") << "Xerces XML parser threw this exception: " << message << std::endl;
   }
   catch (const SAXException &toCatch) {
      char *message = XMLString::transcode(toCatch.getMessage());
      throw cms::Exception("XMLException") << "Xerces XML parser threw this exception: " << message << std::endl;
   }
   catch (...) {
      throw cms::Exception("XMLException") << "Xerces XML parser threw an unknown exception" << std::endl;
   }

   DOMDocument *doc = parser->getDocument();
   DOMElement *node_MuonAlignment = doc->getDocumentElement();
   DOMNodeList *collections = doc->getElementsByTagName(str_collection);
   DOMNodeList *operations = doc->getElementsByTagName(str_operation);

   std::map<Alignable*, Alignable*> alitoideal;
   fillAliToIdeal(alitoideal, alignableMuon->DTBarrel(), ideal_alignableMuon->DTBarrel());
   fillAliToIdeal(alitoideal, alignableMuon->CSCEndcaps(), ideal_alignableMuon->CSCEndcaps());

   std::map<std::string, std::map<Alignable*, bool> > alicollections;
   for (unsigned int i = 0;  i < collections->getLength();  i++) {
      DOMElement *collection = (DOMElement*)(collections->item(i));
      if (collection->getParentNode() == node_MuonAlignment) {
	 DOMNodeList *children = collection->getChildNodes();

	 DOMAttr *node_name = collection->getAttributeNode(str_name);
	 if (node_name == NULL) {
	    throw cms::Exception("XMLException") << "<collection> requires a name attribute" << std::endl;
	 }
	 char *ascii_name = XMLString::transcode(node_name->getValue());
	 std::string name(ascii_name);
	 XMLString::release(&ascii_name);

	 std::map<Alignable*, bool> aliset;
	 for (unsigned int j = 0;  j < children->getLength();  j++) {
	    DOMNode *node = children->item(j);

	    if (node->getNodeType() == DOMNode::ELEMENT_NODE) {
	       Alignable *ali = getNode(alignableNavigator, (DOMElement*)(node));
	       if (ali == NULL) {
		  throw cms::Exception("XMLException") << "<collection> must contain only alignables" << std::endl;
	       }

	       aliset[ali] = true;
	    } // end if this node is an element
	 } // end loop over collection's children

	 alicollections[name] = aliset;
      } // end if this is a top-level collection
   } // end loop over collections

   for (unsigned int i = 0;  i < operations->getLength();  i++) {
      DOMElement *operation = (DOMElement*)(operations->item(i));
      if (operation->getParentNode() != node_MuonAlignment) {
	 throw cms::Exception("XMLException") << "All operations must be top-level elements" << std::endl;
      }

      DOMNodeList *children = operation->getChildNodes();

      std::map<Alignable*, bool> aliset;
      std::vector<DOMNode*> nodesToRemove;
      for (unsigned int j = 0;  j < children->getLength();  j++) {
	 DOMNode *node = children->item(j);

	 if (node->getNodeType() == DOMNode::ELEMENT_NODE) {
	    Alignable *ali = getNode(alignableNavigator, (DOMElement*)(node));
	    if (ali != NULL) {
	       aliset[ali] = true;
	       nodesToRemove.push_back(node);
	    } // end if this node is an alignable

	    else if (XMLString::equals(node->getNodeName(), str_collection)) {
	       DOMAttr *node_name = ((DOMElement*)(node))->getAttributeNode(str_name);
	       if (node_name == NULL) {
		  throw cms::Exception("XMLException") << "<collection> requires a name attribute" << std::endl;
	       }
	       char *ascii_name = XMLString::transcode(node_name->getValue());
	       std::string name(ascii_name);
	       XMLString::release(&ascii_name);

	       std::map<std::string, std::map<Alignable*, bool> >::const_iterator alicollections_iter = alicollections.find(name);
	       if (alicollections_iter == alicollections.end()) {
		  throw cms::Exception("XMLException") << "<collection name=\"" << name << "\"> hasn't been defined" << std::endl;
	       }

	       for (std::map<Alignable*, bool>::const_iterator aliiter = alicollections_iter->second.begin();
		    aliiter != alicollections_iter->second.end();
		    ++aliiter) {
		  aliset[aliiter->first] = true;
	       } // end loop over alignables in this collection

	       nodesToRemove.push_back(node);
	    } // end if this node is a collection

	    else {} // anything else? assume it's a position/rotation directive

	 } // end if node is node is an element
      } // end first loop over operation's children

      // from now on, we only want to see position/rotation directives
      for (std::vector<DOMNode*>::const_iterator node = nodesToRemove.begin();  node != nodesToRemove.end();  ++node) {
	 operation->removeChild(*node);
      }
      children = operation->getChildNodes();

      for (unsigned int j = 0;  j < children->getLength();  j++) {
	 DOMNode *node = children->item(j);
	 if (node->getNodeType() == DOMNode::ELEMENT_NODE) {

	    if (XMLString::equals(node->getNodeName(), str_setposition)) {
	       do_setposition((DOMElement*)(node), aliset, alitoideal);
	    }

	    else if (XMLString::equals(node->getNodeName(), str_setape)) {
	       do_setape((DOMElement*)(node), aliset, alitoideal);
	    }

	    else if (XMLString::equals(node->getNodeName(), str_setsurveyerr)) {
	       do_setsurveyerr((DOMElement*)(node), aliset, alitoideal);
	    }

	    else if (XMLString::equals(node->getNodeName(), str_moveglobal)) {
	       do_moveglobal((DOMElement*)(node), aliset, alitoideal);
	    }

	    else if (XMLString::equals(node->getNodeName(), str_movelocal)) {
	       do_movelocal((DOMElement*)(node), aliset, alitoideal);
	    }

	    else if (XMLString::equals(node->getNodeName(), str_rotatelocal)) {
	       do_rotatelocal((DOMElement*)(node), aliset, alitoideal);
	    }

	    else if (XMLString::equals(node->getNodeName(), str_rotatebeamline)) {
	       do_rotatebeamline((DOMElement*)(node), aliset, alitoideal);
	    }

	    else if (XMLString::equals(node->getNodeName(), str_rotateglobalaxis)) {
	       do_rotateglobalaxis((DOMElement*)(node), aliset, alitoideal);
	    }

	    else {
	       char *message = XMLString::transcode(node->getNodeName());
	       throw cms::Exception("XMLException") << "Unrecognized operation: \"" << message << "\"" << std::endl;
	    }

	 } // end if node is an element
      } // end second loop over operation's children
   } // end loop over operations

   delete parser;
   delete errHandler;

   cms::concurrency::xercesTerminate();

   delete ideal_alignableMuon;
   return alignableMuon;
}

Alignable *MuonAlignmentInputXML::getNode(std::map<unsigned int, Alignable*> &alignableNavigator, const XERCES_CPP_NAMESPACE::DOMElement *node) const {
   if (XMLString::equals(node->getNodeName(), str_DTBarrel)) return getDTnode(align::AlignableDTBarrel, alignableNavigator, node);
   else if (XMLString::equals(node->getNodeName(), str_DTWheel)) return getDTnode(align::AlignableDTWheel, alignableNavigator, node);
   else if (XMLString::equals(node->getNodeName(), str_DTStation)) return getDTnode(align::AlignableDTStation, alignableNavigator, node);
   else if (XMLString::equals(node->getNodeName(), str_DTChamber)) return getDTnode(align::AlignableDTChamber, alignableNavigator, node);
   else if (XMLString::equals(node->getNodeName(), str_DTSuperLayer)) return getDTnode(align::AlignableDTSuperLayer, alignableNavigator, node);
   else if (XMLString::equals(node->getNodeName(), str_DTLayer)) return getDTnode(align::AlignableDetUnit, alignableNavigator, node);
   else if (XMLString::equals(node->getNodeName(), str_CSCEndcap)) return getCSCnode(align::AlignableCSCEndcap, alignableNavigator, node);
   else if (XMLString::equals(node->getNodeName(), str_CSCStation)) return getCSCnode(align::AlignableCSCStation, alignableNavigator, node);
   else if (XMLString::equals(node->getNodeName(), str_CSCRing)) return getCSCnode(align::AlignableCSCRing, alignableNavigator, node);
   else if (XMLString::equals(node->getNodeName(), str_CSCChamber)) return getCSCnode(align::AlignableCSCChamber, alignableNavigator, node);
   else if (XMLString::equals(node->getNodeName(), str_CSCLayer)) return getCSCnode(align::AlignableDetUnit, alignableNavigator, node);
   else return NULL;
}

Alignable *MuonAlignmentInputXML::getDTnode(align::StructureType structureType, std::map<unsigned int, Alignable*> &alignableNavigator, const XERCES_CPP_NAMESPACE::DOMElement *node) const {
   unsigned int rawId = 0;

   DOMAttr *node_rawId = node->getAttributeNode(str_rawId);
   if (node_rawId != NULL) {
      try {
	 rawId = XMLString::parseInt(node_rawId->getValue());
      }
      catch (const XMLException &toCatch) {
	 throw cms::Exception("XMLException") << "Value of \"rawId\" must be an integer" << std::endl;
      }
   }
   else {
      int wheel, station, sector, superlayer, layer;
      wheel = station = sector = superlayer = layer = 1;

      if (structureType != align::AlignableDTBarrel) {
	 DOMAttr *node_wheel = node->getAttributeNode(str_wheel);
	 if (node_wheel == NULL) throw cms::Exception("XMLException") << "DT node is missing required \"wheel\" attribute" << std::endl;
	 try {
	    wheel = XMLString::parseInt(node_wheel->getValue());
	 }
	 catch (const XMLException &toCatch) {
	    throw cms::Exception("XMLException") << "Value of \"wheel\" must be an integer" << std::endl;
	 }

	 if (structureType != align::AlignableDTWheel) {
	    DOMAttr *node_station = node->getAttributeNode(str_station);
	    if (node_station == NULL) throw cms::Exception("XMLException") << "DT node is missing required \"station\" attribute" << std::endl;
	    try {
	       station = XMLString::parseInt(node_station->getValue());
	    }
	    catch (const XMLException &toCatch) {
	       throw cms::Exception("XMLException") << "Value of \"station\" must be an integer" << std::endl;
	    }

	    if (structureType != align::AlignableDTStation) {
	       DOMAttr *node_sector = node->getAttributeNode(str_sector);
	       if (node_sector == NULL) throw cms::Exception("XMLException") << "DT node is missing required \"sector\" attribute" << std::endl;
	       try {
		  sector = XMLString::parseInt(node_sector->getValue());
	       }
	       catch (const XMLException &toCatch) {
		  throw cms::Exception("XMLException") << "Value of \"sector\" must be an integer" << std::endl;
	       }

	       if (structureType != align::AlignableDTChamber) {
		  DOMAttr *node_superlayer = node->getAttributeNode(str_superlayer);
		  if (node_superlayer == NULL) throw cms::Exception("XMLException") << "DT node is missing required \"superlayer\" attribute" << std::endl;
		  try {
		     superlayer = XMLString::parseInt(node_superlayer->getValue());
		  }
		  catch (const XMLException &toCatch) {
		     throw cms::Exception("XMLException") << "Value of \"superlayer\" must be an integer" << std::endl;
		  }

		  if (structureType != align::AlignableDTSuperLayer) {
		     DOMAttr *node_layer = node->getAttributeNode(str_layer);
		     if (node_layer == NULL) throw cms::Exception("XMLException") << "DT node is missing required \"layer\" attribute" << std::endl;
		     try {
			layer = XMLString::parseInt(node_layer->getValue());
		     }
		     catch (const XMLException &toCatch) {
			throw cms::Exception("XMLException") << "Value of \"layer\" must be an integer" << std::endl;
		     }

		  } // end if we need a layer number
	       } // end if we need a superlayer number
	    } // end if we need a sector number
	 } // end if we need a station number
      } // end if we need a wheel number

      DTLayerId layerId(wheel, station, sector, superlayer, layer);
      rawId = layerId.rawId();
   } // end if it's specified by wheel, station, sector, superlayer, layer

   Alignable *ali = alignableNavigator[rawId];
   if (ali == NULL) throw cms::Exception("XMLException") << "rawId \"" << rawId << "\" is not recognized" << std::endl;

   while (ali->alignableObjectId() != structureType) {
      ali = ali->mother();

      if (ali == NULL) {
	 if (structureType == align::AlignableDTBarrel) throw cms::Exception("XMLException") << "rawId \"" << rawId << "\" is not a DTBarrel" << std::endl;
	 else if (structureType == align::AlignableDTWheel) throw cms::Exception("XMLException") << "rawId \"" << rawId << "\" is not a DTWheel" << std::endl;
	 else if (structureType == align::AlignableDTStation) throw cms::Exception("XMLException") << "rawId \"" << rawId << "\" is not a DTStation" << std::endl;
	 else if (structureType == align::AlignableDTChamber) throw cms::Exception("XMLException") << "rawId \"" << rawId << "\" is not a DTChamber" << std::endl;
	 else if (structureType == align::AlignableDTSuperLayer) throw cms::Exception("XMLException") << "rawId \"" << rawId << "\" is not a DTSuperLayer" << std::endl;
	 else if (structureType == align::AlignableDetUnit) throw cms::Exception("XMLException") << "rawId \"" << rawId << "\" is not a DTLayer" << std::endl;
      }
   }
   return ali;
}

Alignable *MuonAlignmentInputXML::getCSCnode(align::StructureType structureType, std::map<unsigned int, Alignable*> &alignableNavigator, const XERCES_CPP_NAMESPACE::DOMElement *node) const {
   unsigned int rawId;

   DOMAttr *node_rawId = node->getAttributeNode(str_rawId);
   if (node_rawId != NULL) {
      try {
	 rawId = XMLString::parseInt(node_rawId->getValue());
      }
      catch (const XMLException &toCatch) {
	 throw cms::Exception("XMLException") << "Value of \"rawId\" must be an integer" << std::endl;
      }
   }
   else {
      int endcap, station, ring, chamber, layer;
      endcap = station = ring = chamber = layer = 1;

      DOMAttr *node_endcap = node->getAttributeNode(str_endcap);
      if (node_endcap == NULL) throw cms::Exception("XMLException") << "CSC node is missing required \"endcap\" attribute" << std::endl;
      try {
	 endcap = XMLString::parseInt(node_endcap->getValue());
      }
      catch (const XMLException &toCatch) {
	 throw cms::Exception("XMLException") << "Value of \"endcap\" must be an integer" << std::endl;
      }
      if (endcap == -1) endcap = 2;

      if (structureType != align::AlignableCSCEndcap) {
	 DOMAttr *node_station = node->getAttributeNode(str_station);
	 if (node_station == NULL) throw cms::Exception("XMLException") << "CSC node is missing required \"station\" attribute" << std::endl;
	 try {
	    station = XMLString::parseInt(node_station->getValue());
	 }
	 catch (const XMLException &toCatch) {
	    throw cms::Exception("XMLException") << "Value of \"station\" must be an integer" << std::endl;
	 }

	 if (structureType != align::AlignableCSCStation) {
	    DOMAttr *node_ring = node->getAttributeNode(str_ring);
	    if (node_ring == NULL) throw cms::Exception("XMLException") << "CSC node is missing required \"ring\" attribute" << std::endl;
	    try {
	       ring = XMLString::parseInt(node_ring->getValue());
	    }
	    catch (const XMLException &toCatch) {
	       throw cms::Exception("XMLException") << "Value of \"ring\" must be an integer" << std::endl;
	    }

	    if (structureType != align::AlignableCSCRing) {
	       DOMAttr *node_chamber = node->getAttributeNode(str_chamber);
	       if (node_chamber == NULL) throw cms::Exception("XMLException") << "CSC node is missing required \"chamber\" attribute" << std::endl;
	       try {
		  chamber = XMLString::parseInt(node_chamber->getValue());
	       }
	       catch (const XMLException &toCatch) {
		  throw cms::Exception("XMLException") << "Value of \"chamber\" must be an integer" << std::endl;
	       }

	       if (structureType != align::AlignableCSCChamber) {
		  DOMAttr *node_layer = node->getAttributeNode(str_layer);
		  if (node_layer == NULL) throw cms::Exception("XMLException") << "CSC node is missing required \"layer\" attribute" << std::endl;
		  try {
		     layer = XMLString::parseInt(node_layer->getValue());
		  }
		  catch (const XMLException &toCatch) {
		     throw cms::Exception("XMLException") << "Value of \"layer\" must be an integer" << std::endl;
		  }

	       } // end if we need a layer number
	    } // end if we need a chamber number
	 } // end if we need a ring number
      } // end if we need a station number

      CSCDetId layerId(endcap, station, ring, chamber, layer);
      rawId = layerId.rawId();
   } // end if it's specified by endcap, station, ring, chamber, layer

   Alignable *ali = alignableNavigator[rawId];
   if (ali == NULL) throw cms::Exception("XMLException") << "rawId \"" << rawId << "\" is not recognized" << std::endl;

   while (ali->alignableObjectId() != structureType) {
      ali = ali->mother();

      if (ali == NULL) {
	 if (structureType == align::AlignableCSCEndcap) throw cms::Exception("XMLException") << "rawId \"" << rawId << "\" is not a CSCEndcap" << std::endl;
	 else if (structureType == align::AlignableCSCStation) throw cms::Exception("XMLException") << "rawId \"" << rawId << "\" is not a CSCStation" << std::endl;
	 else if (structureType == align::AlignableCSCRing) throw cms::Exception("XMLException") << "rawId \"" << rawId << "\" is not a CSCRing" << std::endl;
	 else if (structureType == align::AlignableCSCChamber) throw cms::Exception("XMLException") << "rawId \"" << rawId << "\" is not a CSCChamber" << std::endl;
	 else if (structureType == align::AlignableDetUnit) throw cms::Exception("XMLException") << "rawId \"" << rawId << "\" is not a CSCLayer" << std::endl;
      }
   }
   return ali;
}

double MuonAlignmentInputXML::parseDouble(const XMLCh *str, const char *attribute) const {
   unsigned int len = XMLString::stringLen(str);
   char *cstr = XMLString::transcode(str);
   std::stringstream errmessage;
   errmessage << "Value of \"" << attribute << "\" must be a double, not \"" << cstr << "\"" << std::endl;

   unsigned int i = 0;

   bool minus = false;
   if (cstr[i] == '-') {
      minus = true;
      i++;
   }
   else if (cstr[i] == '+') i++;

   double output = 0.;

   while (cstr[i] != '.'  &&  cstr[i] != 'e'  &&  cstr[i] != 'E'  &&  i < len) {
      if (cstr[i] < '0'  ||  cstr[i] > '9') {
	 XMLString::release(&cstr);
	 throw cms::Exception("XMLException") << errmessage.str();	 
      }

      output *= 10;
      output += cstr[i] - '0';
      i++;
   }

   if (cstr[i] == '.') {
      double place = 0.1;
      i++;

      while (cstr[i] != 'e'  &&  cstr[i] != 'E'  &&  i < len) {
	 if (cstr[i] < '0'  ||  cstr[i] > '9') {
	    XMLString::release(&cstr);
	    throw cms::Exception("XMLException") << errmessage.str();	 
	 }
	 
	 output += (cstr[i] - '0') * place;
	 place /= 10.;
	 i++;
      }
   }

   if (cstr[i] == 'e'  ||  cstr[i] == 'E') {
      i++;

      int exponent = 0;
      bool expminus = false;
      if (cstr[i] == '-') {
	 expminus = true;
	 i++;
      }
      else if (cstr[i] == '+') i++;

      while (i < len) {
	 if (cstr[i] < '0'  ||  cstr[i] > '9') {
	    XMLString::release(&cstr);
	    throw cms::Exception("XMLException") << errmessage.str();	 
	 }
      
	 exponent *= 10;
	 exponent += cstr[i] - '0';
	 i++;
      }

      if (expminus) exponent *= -1;

      output *= pow(10., exponent);
   }

   if (minus) output *= -1.;

   XMLString::release(&cstr);
   return output;
}

void MuonAlignmentInputXML::do_setposition(const XERCES_CPP_NAMESPACE::DOMElement *node, std::map<Alignable*, bool> &aliset, std::map<Alignable*, Alignable*> &alitoideal) const {
  DOMAttr *node_relativeto = node->getAttributeNode(str_relativeto);
  if (node_relativeto == NULL) throw cms::Exception("XMLException") << "<setposition> is missing required \"relativeto\" attribute" << std::endl;

  int relativeto = 0;
  if (XMLString::equals(node_relativeto->getValue(), str_none)) {
    relativeto = 0;
  }
  else if (XMLString::equals(node_relativeto->getValue(), str_ideal)) {
    relativeto = 1;
  }
  else if (XMLString::equals(node_relativeto->getValue(), str_container)) {
    relativeto = 2;
  }
  else {
    char *message = XMLString::transcode(node_relativeto->getValue());
    throw cms::Exception("XMLException") << "relativeto must be \"none\", \"ideal\", or \"container\", not \"" << message << "\"" << std::endl;
  }

  for (std::map<Alignable*, bool>::const_iterator aliiter = aliset.begin();  aliiter != aliset.end();  ++aliiter) {
    // first reconstruct the old position: how it would look in this coordinate system

    Alignable *ali = aliiter->first;
    Alignable *ideal = alitoideal[ali];

    align::PositionType oldpos = ali->globalPosition();
    align::RotationType oldrot = ali->globalRotation();

    if (relativeto == 0) {}

    else if (relativeto == 1) {
      align::PositionType idealPosition = ideal->globalPosition();
      align::RotationType idealRotation = ideal->globalRotation();

      oldpos = align::PositionType(idealRotation * (oldpos.basicVector() - idealPosition.basicVector()));
      oldrot = oldrot * idealRotation.transposed();
    }

    else if (relativeto == 2  &&  ali->mother() != NULL) {
      align::PositionType globalPosition = ali->mother()->globalPosition();
      align::RotationType globalRotation = ali->mother()->globalRotation();

      oldpos = align::PositionType(globalRotation * (oldpos.basicVector() - globalPosition.basicVector()));
      oldrot = oldrot * globalRotation.transposed();
    }

    double x = oldpos.x();
    double y = oldpos.y();
    double z = oldpos.z();

    double phix = atan2(oldrot.yz(), oldrot.zz());
    double phiy = asin(-oldrot.xz());
    double phiz = atan2(oldrot.xy(), oldrot.xx());

    align::EulerAngles oldEulerAngles = align::toAngles(oldrot);
    double alpha = oldEulerAngles(1);
    double beta = oldEulerAngles(2);
    double gamma = oldEulerAngles(3);

    // now get the new information; if it's incomplete, use the old position for those coordinates
    
    DOMAttr *node_x = node->getAttributeNode(str_x);
    DOMAttr *node_y = node->getAttributeNode(str_y);
    DOMAttr *node_z = node->getAttributeNode(str_z);

    if (node_x != NULL) x = parseDouble(node_x->getValue(), "x");
    if (node_y != NULL) y = parseDouble(node_y->getValue(), "y");
    if (node_z != NULL) z = parseDouble(node_z->getValue(), "z");
    align::PositionType pos(x, y, z);

    DOMAttr *node_phix = node->getAttributeNode(str_phix);
    DOMAttr *node_phiy = node->getAttributeNode(str_phiy);
    DOMAttr *node_phiz = node->getAttributeNode(str_phiz);
    DOMAttr *node_alpha = node->getAttributeNode(str_alpha);
    DOMAttr *node_beta  = node->getAttributeNode(str_beta);
    DOMAttr *node_gamma = node->getAttributeNode(str_gamma);
    align::RotationType rot;

    bool phixyz = (node_phix != NULL  ||  node_phiy != NULL  ||  node_phiz != NULL);
    bool alphabetagamma = (node_alpha != NULL  ||  node_beta != NULL  ||  node_gamma != NULL);
    if (phixyz && alphabetagamma) throw cms::Exception("XMLException") << "<setposition> must either have phix, phiy, and phiz or alpha, beta, and gamma, but not both" << std::endl;
    if (!phixyz && !alphabetagamma) alphabetagamma = true;

    if (phixyz) {
      if (node_phix != NULL) phix = parseDouble(node_phix->getValue(), "phix");
      if (node_phiy != NULL) phiy = parseDouble(node_phiy->getValue(), "phiy");
      if (node_phiz != NULL) phiz = parseDouble(node_phiz->getValue(), "phiz");

      // the angle convention originally used in alignment, also known as "non-standard Euler angles with a Z-Y-X convention"
      // this also gets the sign convention right
      align::RotationType rotX( 1.,         0.,         0.,
				0.,         cos(phix),  sin(phix),
				0.,        -sin(phix),  cos(phix));
      align::RotationType rotY( cos(phiy),  0.,        -sin(phiy), 
				0.,         1.,         0.,
				sin(phiy),  0.,         cos(phiy));
      align::RotationType rotZ( cos(phiz),  sin(phiz),  0.,
				-sin(phiz),  cos(phiz),  0.,
				0.,         0.,         1.);
            
      rot = rotX * rotY * rotZ;
    }

    else if (alphabetagamma) {
      if (node_alpha != NULL) alpha = parseDouble(node_alpha->getValue(), "alpha");
      if (node_beta != NULL) beta = parseDouble(node_beta->getValue(), "beta");
      if (node_gamma != NULL) gamma = parseDouble(node_gamma->getValue(), "gamma");

      // standard Euler angles (how they're internally stored in the database)
      align::EulerAngles eulerAngles(3);
      eulerAngles(1) = alpha;
      eulerAngles(2) = beta;
      eulerAngles(3) = gamma;
      rot = align::RotationType(align::toMatrix(eulerAngles));
    }

    else assert(false); // see above

    if (relativeto == 0) {
      set_one_position(aliiter->first, pos, rot);
    } // end relativeto="none"

    else if (relativeto == 1) {
      Alignable *ali = aliiter->first;
      Alignable *ideal = alitoideal[ali];

      align::PositionType idealPosition = ideal->globalPosition();
      align::RotationType idealRotation = ideal->globalRotation();
      align::PositionType newpos = align::PositionType(idealRotation.transposed() * pos.basicVector() + idealPosition.basicVector());
      align::RotationType newrot = rot * idealRotation;

      set_one_position(ali, newpos, newrot);
    } // end relativeto="ideal"

    else if (relativeto == 2) {
      Alignable *ali = aliiter->first;
      Alignable *container = ali->mother();

      if (container != NULL) {
	align::PositionType globalPosition = container->globalPosition();
	align::RotationType globalRotation = container->globalRotation();
	align::PositionType newpos = align::PositionType(globalRotation.transposed() * pos.basicVector() + globalPosition.basicVector());
	align::RotationType newrot = rot * globalRotation;
	set_one_position(ali, newpos, newrot);
      }
      else {
	set_one_position(ali, pos, rot);
      }
    } // end relativeto="container"

  } // end loop over alignables
}

void MuonAlignmentInputXML::set_one_position(Alignable *ali, const align::PositionType &pos, const align::RotationType &rot) const {
   const align::PositionType& oldpos = ali->globalPosition();
   const align::RotationType& oldrot = ali->globalRotation();
                                 
   // shift needed to move from current to new position
   align::GlobalVector posDiff = pos - oldpos;
   align::RotationType rotDiff = oldrot.multiplyInverse(rot);
   align::rectify(rotDiff); // correct for rounding errors 
   ali->move(posDiff);
   ali->rotateInGlobalFrame(rotDiff);

//    // check for consistency
//    const align::PositionType& newpos = ali->globalPosition();
//    const align::RotationType& newrot = ali->globalRotation();
//    align::GlobalVector posDiff2 = pos - newpos;
//    align::RotationType rotDiff2 = newrot.multiplyInverse(rot);
//    align::rectify(rotDiff2); // correct for rounding errors 
   
//    if (fabs(posDiff2.x()) > 1e-6  ||  fabs(posDiff2.y()) > 1e-6  ||  fabs(posDiff2.z()) > 1e-6) {
//       std::cout << "zeropos " << posDiff2 << std::endl;
//    }
//    if (fabs(rotDiff2.xx() - 1.) > 1e-4  ||
//        fabs(rotDiff2.yy() - 1.) > 1e-4  ||
//        fabs(rotDiff2.zz() - 1.) > 1e-4  ||
//        fabs(rotDiff2.xy()) > 1e-8  ||
//        fabs(rotDiff2.xz()) > 1e-8  ||
//        fabs(rotDiff2.yz()) > 1e-8) {
//       std::cout << "zerorot " << rotDiff2 << std::endl;
//    }

   align::ErrorMatrix matrix6x6 = ROOT::Math::SMatrixIdentity();
   matrix6x6 *= 1000.;  // initial assumption: infinitely weak constraint

   const SurveyDet *survey = ali->survey();
   if (survey != NULL) {
      matrix6x6 = survey->errors();  // save the constraint information
   }
   ali->setSurvey(new SurveyDet(ali->surface(), matrix6x6));
}

void MuonAlignmentInputXML::do_setape(const XERCES_CPP_NAMESPACE::DOMElement *node, std::map<Alignable*, bool> &aliset, std::map<Alignable*, Alignable*> &alitoideal) const {
   DOMAttr *node_xx = node->getAttributeNode(str_xx);
   DOMAttr *node_xy = node->getAttributeNode(str_xy);
   DOMAttr *node_xz = node->getAttributeNode(str_xz);
   DOMAttr *node_xa = node->getAttributeNode(str_xa);
   DOMAttr *node_xb = node->getAttributeNode(str_xb);
   DOMAttr *node_xc = node->getAttributeNode(str_xc);
   DOMAttr *node_yy = node->getAttributeNode(str_yy);
   DOMAttr *node_yz = node->getAttributeNode(str_yz);
   DOMAttr *node_ya = node->getAttributeNode(str_ya);
   DOMAttr *node_yb = node->getAttributeNode(str_yb);
   DOMAttr *node_yc = node->getAttributeNode(str_yc);
   DOMAttr *node_zz = node->getAttributeNode(str_zz);
   DOMAttr *node_za = node->getAttributeNode(str_za);
   DOMAttr *node_zb = node->getAttributeNode(str_zb);
   DOMAttr *node_zc = node->getAttributeNode(str_zc);
   DOMAttr *node_aa = node->getAttributeNode(str_aa);
   DOMAttr *node_ab = node->getAttributeNode(str_ab);
   DOMAttr *node_ac = node->getAttributeNode(str_ac);
   DOMAttr *node_bb = node->getAttributeNode(str_bb);
   DOMAttr *node_bc = node->getAttributeNode(str_bc);
   DOMAttr *node_cc = node->getAttributeNode(str_cc);

   if (node_xx == NULL) throw cms::Exception("XMLException") << "<setape> is missing required \"xx\" attribute" << std::endl;
   if (node_xy == NULL) throw cms::Exception("XMLException") << "<setape> is missing required \"xy\" attribute" << std::endl;
   if (node_xz == NULL) throw cms::Exception("XMLException") << "<setape> is missing required \"xz\" attribute" << std::endl;
   if (node_xa == NULL) throw cms::Exception("XMLException") << "<setape> is missing required \"xa\" attribute" << std::endl;
   if (node_xb == NULL) throw cms::Exception("XMLException") << "<setape> is missing required \"xb\" attribute" << std::endl;
   if (node_xc == NULL) throw cms::Exception("XMLException") << "<setape> is missing required \"xc\" attribute" << std::endl;
   if (node_yy == NULL) throw cms::Exception("XMLException") << "<setape> is missing required \"yy\" attribute" << std::endl;
   if (node_yz == NULL) throw cms::Exception("XMLException") << "<setape> is missing required \"yz\" attribute" << std::endl;
   if (node_ya == NULL) throw cms::Exception("XMLException") << "<setape> is missing required \"ya\" attribute" << std::endl;
   if (node_yb == NULL) throw cms::Exception("XMLException") << "<setape> is missing required \"yb\" attribute" << std::endl;
   if (node_yc == NULL) throw cms::Exception("XMLException") << "<setape> is missing required \"yc\" attribute" << std::endl;
   if (node_zz == NULL) throw cms::Exception("XMLException") << "<setape> is missing required \"zz\" attribute" << std::endl;
   if (node_za == NULL) throw cms::Exception("XMLException") << "<setape> is missing required \"za\" attribute" << std::endl;
   if (node_zb == NULL) throw cms::Exception("XMLException") << "<setape> is missing required \"zb\" attribute" << std::endl;
   if (node_zc == NULL) throw cms::Exception("XMLException") << "<setape> is missing required \"zc\" attribute" << std::endl;
   if (node_aa == NULL) throw cms::Exception("XMLException") << "<setape> is missing required \"aa\" attribute" << std::endl;
   if (node_ab == NULL) throw cms::Exception("XMLException") << "<setape> is missing required \"ab\" attribute" << std::endl;
   if (node_ac == NULL) throw cms::Exception("XMLException") << "<setape> is missing required \"ac\" attribute" << std::endl;
   if (node_bb == NULL) throw cms::Exception("XMLException") << "<setape> is missing required \"bb\" attribute" << std::endl;
   if (node_bc == NULL) throw cms::Exception("XMLException") << "<setape> is missing required \"bc\" attribute" << std::endl;
   if (node_cc == NULL) throw cms::Exception("XMLException") << "<setape> is missing required \"cc\" attribute" << std::endl;

   align::ErrorMatrix matrix6x6;
   matrix6x6(0,0) = parseDouble(node_xx->getValue(), "xx");
   matrix6x6(0,1) = parseDouble(node_xy->getValue(), "xy");
   matrix6x6(0,2) = parseDouble(node_xz->getValue(), "xz");
   matrix6x6(0,3) = parseDouble(node_xa->getValue(), "xa");
   matrix6x6(0,4) = parseDouble(node_xb->getValue(), "xb");
   matrix6x6(0,5) = parseDouble(node_xc->getValue(), "xc");
   matrix6x6(1,1) = parseDouble(node_yy->getValue(), "yy");
   matrix6x6(1,2) = parseDouble(node_yz->getValue(), "yz");
   matrix6x6(1,3) = parseDouble(node_ya->getValue(), "ya");
   matrix6x6(1,4) = parseDouble(node_yb->getValue(), "yb");
   matrix6x6(1,5) = parseDouble(node_yc->getValue(), "yc");
   matrix6x6(2,2) = parseDouble(node_zz->getValue(), "zz");
   matrix6x6(2,3) = parseDouble(node_za->getValue(), "za");
   matrix6x6(2,4) = parseDouble(node_zb->getValue(), "zb");
   matrix6x6(2,5) = parseDouble(node_zc->getValue(), "zc");
   matrix6x6(3,3) = parseDouble(node_aa->getValue(), "aa");
   matrix6x6(3,4) = parseDouble(node_ab->getValue(), "ab");
   matrix6x6(3,5) = parseDouble(node_ac->getValue(), "ac");
   matrix6x6(4,4) = parseDouble(node_bb->getValue(), "bb");
   matrix6x6(4,5) = parseDouble(node_bc->getValue(), "bc");
   matrix6x6(5,5) = parseDouble(node_cc->getValue(), "cc");

   for (std::map<Alignable*, bool>::const_iterator aliiter = aliset.begin();  aliiter != aliset.end();  ++aliiter) {
     // this sets APEs at this level and (since 2nd argument is true) all lower levels
     aliiter->first->setAlignmentPositionError(AlignmentPositionError(matrix6x6), true);
   }
}

void MuonAlignmentInputXML::do_setsurveyerr(const XERCES_CPP_NAMESPACE::DOMElement *node, std::map<Alignable*, bool> &aliset, std::map<Alignable*, Alignable*> &alitoideal) const {
   DOMAttr *node_xx = node->getAttributeNode(str_xx);
   DOMAttr *node_xy = node->getAttributeNode(str_xy);
   DOMAttr *node_xz = node->getAttributeNode(str_xz);
   DOMAttr *node_xa = node->getAttributeNode(str_xa);
   DOMAttr *node_xb = node->getAttributeNode(str_xb);
   DOMAttr *node_xc = node->getAttributeNode(str_xc);
   DOMAttr *node_yy = node->getAttributeNode(str_yy);
   DOMAttr *node_yz = node->getAttributeNode(str_yz);
   DOMAttr *node_ya = node->getAttributeNode(str_ya);
   DOMAttr *node_yb = node->getAttributeNode(str_yb);
   DOMAttr *node_yc = node->getAttributeNode(str_yc);
   DOMAttr *node_zz = node->getAttributeNode(str_zz);
   DOMAttr *node_za = node->getAttributeNode(str_za);
   DOMAttr *node_zb = node->getAttributeNode(str_zb);
   DOMAttr *node_zc = node->getAttributeNode(str_zc);
   DOMAttr *node_aa = node->getAttributeNode(str_aa);
   DOMAttr *node_ab = node->getAttributeNode(str_ab);
   DOMAttr *node_ac = node->getAttributeNode(str_ac);
   DOMAttr *node_bb = node->getAttributeNode(str_bb);
   DOMAttr *node_bc = node->getAttributeNode(str_bc);
   DOMAttr *node_cc = node->getAttributeNode(str_cc);

   if (node_xx == NULL) throw cms::Exception("XMLException") << "<setsurveyerr> is missing required \"xx\" attribute" << std::endl;
   if (node_xy == NULL) throw cms::Exception("XMLException") << "<setsurveyerr> is missing required \"xy\" attribute" << std::endl;
   if (node_xz == NULL) throw cms::Exception("XMLException") << "<setsurveyerr> is missing required \"xz\" attribute" << std::endl;
   if (node_xa == NULL) throw cms::Exception("XMLException") << "<setsurveyerr> is missing required \"xa\" attribute" << std::endl;
   if (node_xb == NULL) throw cms::Exception("XMLException") << "<setsurveyerr> is missing required \"xb\" attribute" << std::endl;
   if (node_xc == NULL) throw cms::Exception("XMLException") << "<setsurveyerr> is missing required \"xc\" attribute" << std::endl;
   if (node_yy == NULL) throw cms::Exception("XMLException") << "<setsurveyerr> is missing required \"yy\" attribute" << std::endl;
   if (node_yz == NULL) throw cms::Exception("XMLException") << "<setsurveyerr> is missing required \"yz\" attribute" << std::endl;
   if (node_ya == NULL) throw cms::Exception("XMLException") << "<setsurveyerr> is missing required \"ya\" attribute" << std::endl;
   if (node_yb == NULL) throw cms::Exception("XMLException") << "<setsurveyerr> is missing required \"yb\" attribute" << std::endl;
   if (node_yc == NULL) throw cms::Exception("XMLException") << "<setsurveyerr> is missing required \"yc\" attribute" << std::endl;
   if (node_zz == NULL) throw cms::Exception("XMLException") << "<setsurveyerr> is missing required \"zz\" attribute" << std::endl;
   if (node_za == NULL) throw cms::Exception("XMLException") << "<setsurveyerr> is missing required \"za\" attribute" << std::endl;
   if (node_zb == NULL) throw cms::Exception("XMLException") << "<setsurveyerr> is missing required \"zb\" attribute" << std::endl;
   if (node_zc == NULL) throw cms::Exception("XMLException") << "<setsurveyerr> is missing required \"zc\" attribute" << std::endl;
   if (node_aa == NULL) throw cms::Exception("XMLException") << "<setsurveyerr> is missing required \"aa\" attribute" << std::endl;
   if (node_ab == NULL) throw cms::Exception("XMLException") << "<setsurveyerr> is missing required \"ab\" attribute" << std::endl;
   if (node_ac == NULL) throw cms::Exception("XMLException") << "<setsurveyerr> is missing required \"ac\" attribute" << std::endl;
   if (node_bb == NULL) throw cms::Exception("XMLException") << "<setsurveyerr> is missing required \"bb\" attribute" << std::endl;
   if (node_bc == NULL) throw cms::Exception("XMLException") << "<setsurveyerr> is missing required \"bc\" attribute" << std::endl;
   if (node_cc == NULL) throw cms::Exception("XMLException") << "<setsurveyerr> is missing required \"cc\" attribute" << std::endl;

   align::ErrorMatrix matrix6x6;
   matrix6x6(0,0) = parseDouble(node_xx->getValue(), "xx");
   matrix6x6(0,1) = parseDouble(node_xy->getValue(), "xy");
   matrix6x6(0,2) = parseDouble(node_xz->getValue(), "xz");
   matrix6x6(0,3) = parseDouble(node_xa->getValue(), "xa");
   matrix6x6(0,4) = parseDouble(node_xb->getValue(), "xb");
   matrix6x6(0,5) = parseDouble(node_xc->getValue(), "xc");
   matrix6x6(1,1) = parseDouble(node_yy->getValue(), "yy");
   matrix6x6(1,2) = parseDouble(node_yz->getValue(), "yz");
   matrix6x6(1,3) = parseDouble(node_ya->getValue(), "ya");
   matrix6x6(1,4) = parseDouble(node_yb->getValue(), "yb");
   matrix6x6(1,5) = parseDouble(node_yc->getValue(), "yc");
   matrix6x6(2,2) = parseDouble(node_zz->getValue(), "zz");
   matrix6x6(2,3) = parseDouble(node_za->getValue(), "za");
   matrix6x6(2,4) = parseDouble(node_zb->getValue(), "zb");
   matrix6x6(2,5) = parseDouble(node_zc->getValue(), "zc");
   matrix6x6(3,3) = parseDouble(node_aa->getValue(), "aa");
   matrix6x6(3,4) = parseDouble(node_ab->getValue(), "ab");
   matrix6x6(3,5) = parseDouble(node_ac->getValue(), "ac");
   matrix6x6(4,4) = parseDouble(node_bb->getValue(), "bb");
   matrix6x6(4,5) = parseDouble(node_bc->getValue(), "bc");
   matrix6x6(5,5) = parseDouble(node_cc->getValue(), "cc");

   for (std::map<Alignable*, bool>::const_iterator aliiter = aliset.begin();  aliiter != aliset.end();  ++aliiter) {
      Alignable *ali = aliiter->first;
      ali->setSurvey(new SurveyDet(ali->surface(), matrix6x6));
   }
}

void MuonAlignmentInputXML::do_moveglobal(const XERCES_CPP_NAMESPACE::DOMElement *node, std::map<Alignable*, bool> &aliset, std::map<Alignable*, Alignable*> &alitoideal) const {
   DOMAttr *node_x = node->getAttributeNode(str_x);
   DOMAttr *node_y = node->getAttributeNode(str_y);
   DOMAttr *node_z = node->getAttributeNode(str_z);
   if (node_x == NULL) throw cms::Exception("XMLException") << "<moveglobal> is missing required \"x\" attribute" << std::endl;
   if (node_y == NULL) throw cms::Exception("XMLException") << "<moveglobal> is missing required \"y\" attribute" << std::endl;
   if (node_z == NULL) throw cms::Exception("XMLException") << "<moveglobal> is missing required \"z\" attribute" << std::endl;

   double x = parseDouble(node_x->getValue(), "x");
   double y = parseDouble(node_y->getValue(), "y");
   double z = parseDouble(node_z->getValue(), "z");
   align::GlobalVector vect(x, y, z);

   for (std::map<Alignable*, bool>::const_iterator aliiter = aliset.begin();  aliiter != aliset.end();  ++aliiter) {
      Alignable *ali = aliiter->first;

      ali->move(vect);

      align::ErrorMatrix matrix6x6 = ROOT::Math::SMatrixIdentity();
      matrix6x6 *= 1000.;  // initial assumption: infinitely weak constraint

      const SurveyDet *survey = ali->survey();
      if (survey != NULL) {
	 matrix6x6 = survey->errors();  // save the constraint information
      }
      ali->setSurvey(new SurveyDet(ali->surface(), matrix6x6));
   } // end loop over alignables
}

void MuonAlignmentInputXML::do_movelocal(const XERCES_CPP_NAMESPACE::DOMElement *node, std::map<Alignable*, bool> &aliset, std::map<Alignable*, Alignable*> &alitoideal) const {
   DOMAttr *node_x = node->getAttributeNode(str_x);
   DOMAttr *node_y = node->getAttributeNode(str_y);
   DOMAttr *node_z = node->getAttributeNode(str_z);
   if (node_x == NULL) throw cms::Exception("XMLException") << "<movelocal> is missing required \"x\" attribute" << std::endl;
   if (node_y == NULL) throw cms::Exception("XMLException") << "<movelocal> is missing required \"y\" attribute" << std::endl;
   if (node_z == NULL) throw cms::Exception("XMLException") << "<movelocal> is missing required \"z\" attribute" << std::endl;

   double x = parseDouble(node_x->getValue(), "x");
   double y = parseDouble(node_y->getValue(), "y");
   double z = parseDouble(node_z->getValue(), "z");
   align::LocalVector vect(x, y, z);

   for (std::map<Alignable*, bool>::const_iterator aliiter = aliset.begin();  aliiter != aliset.end();  ++aliiter) {
      Alignable *ali = aliiter->first;

      align::GlobalVector globalVector = ali->surface().toGlobal(vect);
      ali->move(globalVector);

      align::ErrorMatrix matrix6x6 = ROOT::Math::SMatrixIdentity();
      matrix6x6 *= 1000.;  // initial assumption: infinitely weak constraint

      const SurveyDet *survey = ali->survey();
      if (survey != NULL) {
	 matrix6x6 = survey->errors();  // save the constraint information
      }
      ali->setSurvey(new SurveyDet(ali->surface(), matrix6x6));
   } // end loop over alignables
}

void MuonAlignmentInputXML::do_rotatelocal(const XERCES_CPP_NAMESPACE::DOMElement *node, std::map<Alignable*, bool> &aliset, std::map<Alignable*, Alignable*> &alitoideal) const {
   DOMAttr *node_axisx = node->getAttributeNode(str_axisx);
   DOMAttr *node_axisy = node->getAttributeNode(str_axisy);
   DOMAttr *node_axisz = node->getAttributeNode(str_axisz);
   DOMAttr *node_angle = node->getAttributeNode(str_angle);
   if (node_axisx == NULL) throw cms::Exception("XMLException") << "<rotatelocal> is missing required \"axisx\" attribute" << std::endl;
   if (node_axisy == NULL) throw cms::Exception("XMLException") << "<rotatelocal> is missing required \"axisy\" attribute" << std::endl;
   if (node_axisz == NULL) throw cms::Exception("XMLException") << "<rotatelocal> is missing required \"axisz\" attribute" << std::endl;
   if (node_angle == NULL) throw cms::Exception("XMLException") << "<rotatelocal> is missing required \"angle\" attribute" << std::endl;

   double x = parseDouble(node_axisx->getValue(), "x");
   double y = parseDouble(node_axisy->getValue(), "y");
   double z = parseDouble(node_axisz->getValue(), "z");
   double angle = parseDouble(node_angle->getValue(), "angle");
   align::LocalVector vect(x, y, z);

   for (std::map<Alignable*, bool>::const_iterator aliiter = aliset.begin();  aliiter != aliset.end();  ++aliiter) {
      Alignable *ali = aliiter->first;

      ali->rotateAroundLocalAxis(vect, angle);

      align::ErrorMatrix matrix6x6 = ROOT::Math::SMatrixIdentity();
      matrix6x6 *= 1000.;  // initial assumption: infinitely weak constraint

      const SurveyDet *survey = ali->survey();
      if (survey != NULL) {
	 matrix6x6 = survey->errors();  // save the constraint information
      }
      ali->setSurvey(new SurveyDet(ali->surface(), matrix6x6));
   } // end loop over alignables
}

void MuonAlignmentInputXML::do_rotatebeamline(const XERCES_CPP_NAMESPACE::DOMElement *node, std::map<Alignable*, bool> &aliset, std::map<Alignable*, Alignable*> &alitoideal) const {
   DOMAttr *node_rphi = node->getAttributeNode(str_rphi);
   DOMAttr *node_phi = node->getAttributeNode(str_phi);
   if (node_rphi == NULL  &&  node_phi == NULL) throw cms::Exception("XMLException") << "<rotatebeamline> is missing required \"*phi\" attribute" << std::endl;
   if (node_rphi != NULL  &&  node_phi != NULL) throw cms::Exception("XMLException") << "<rotatebeamline> can't have both an \"rphi\" and a \"phi\" attribute" << std::endl;

   double value;
   if (node_rphi != NULL) {
      value = parseDouble(node_rphi->getValue(), "rphi");
   }  
   else {
      value = parseDouble(node_phi->getValue(), "phi");
   }  

   for (std::map<Alignable*, bool>::const_iterator aliiter = aliset.begin();  aliiter != aliset.end();  ++aliiter) {
      Alignable *ali = aliiter->first;

      align::GlobalPoint pos = ali->surface().toGlobal(align::LocalPoint(0,0,0));

      double radius = pos.perp();
      double phi0 = pos.phi();
      double deltaphi = value;
      if (node_rphi != NULL) deltaphi = value / radius;

      ali->rotateAroundGlobalZ(deltaphi);
      ali->move(align::GlobalVector(radius * (cos(phi0 + deltaphi) - cos(phi0)),
                                    radius * (sin(phi0 + deltaphi) - sin(phi0)),
                                    0.));

      align::ErrorMatrix matrix6x6 = ROOT::Math::SMatrixIdentity();
      matrix6x6 *= 1000.;  // initial assumption: infinitely weak constraint

      const SurveyDet *survey = ali->survey();
      if (survey != NULL) {
         matrix6x6 = survey->errors();  // save the constraint information
      }
      ali->setSurvey(new SurveyDet(ali->surface(), matrix6x6));
   } // end loop over alignables
}

void MuonAlignmentInputXML::do_rotateglobalaxis(const XERCES_CPP_NAMESPACE::DOMElement *node, std::map<Alignable*, bool> &aliset, std::map<Alignable*, Alignable*> &alitoideal) const {
   DOMAttr *node_x = node->getAttributeNode(str_x);
   DOMAttr *node_y = node->getAttributeNode(str_y);
   DOMAttr *node_z = node->getAttributeNode(str_z);
   DOMAttr *node_angle = node->getAttributeNode(str_angle);
   if (node_x == NULL) throw cms::Exception("XMLException") << "<rotateglobalaxis> is missing required \"x\" attribute" << std::endl;
   if (node_y == NULL) throw cms::Exception("XMLException") << "<rotateglobalaxis> is missing required \"y\" attribute" << std::endl;
   if (node_z == NULL) throw cms::Exception("XMLException") << "<rotateglobalaxis> is missing required \"z\" attribute" << std::endl;
   if (node_angle == NULL) throw cms::Exception("XMLException") << "<rotateglobalaxis> is missing required \"angle\" attribute" << std::endl;

   double x = parseDouble(node_x->getValue(), "x");
   double y = parseDouble(node_y->getValue(), "y");
   double z = parseDouble(node_z->getValue(), "z");
   double angle = parseDouble(node_angle->getValue(), "angle");

   for (std::map<Alignable*, bool>::const_iterator aliiter = aliset.begin();  aliiter != aliset.end();  ++aliiter) {
      Alignable *ali = aliiter->first;
      align::GlobalPoint pos = ali->surface().toGlobal(align::LocalPoint(0,0,0));

      ali->rotateAroundGlobalAxis(align::GlobalVector(x, y, z), angle);

      double aprime = x/sqrt(x*x + y*y + z*z);
      double bprime = y/sqrt(x*x + y*y + z*z);
      double cprime = z/sqrt(x*x + y*y + z*z);
      double q0 = cos(angle/2.);
      double q1 = sin(angle/2.) * aprime;
      double q2 = sin(angle/2.) * bprime;
      double q3 = sin(angle/2.) * cprime;
      
      double pos2x = (q0*q0 + q1*q1 - q2*q2 - q3*q3) * pos.x()  +  2.*(q1*q2 - q0*q3) * pos.y()  +  2.*(q1*q3 + q0*q2) * pos.z();
      double pos2y = 2.*(q2*q1 + q0*q3) * pos.x()  +  (q0*q0 - q1*q1 + q2*q2 - q3*q3) * pos.y()  +  2.*(q2*q3 - q0*q1) * pos.z();
      double pos2z = 2.*(q3*q1 - q0*q2) * pos.x()  +  2.*(q3*q2 + q0*q1) * pos.y()  +  (q0*q0 - q1*q1 - q2*q2 + q3*q3) * pos.z();

      double movex = pos2x - pos.x();
      double movey = pos2y - pos.y();
      double movez = pos2z - pos.z();
      ali->move(align::GlobalVector(movex, movey, movez));

      align::ErrorMatrix matrix6x6 = ROOT::Math::SMatrixIdentity();
      matrix6x6 *= 1000.;  // initial assumption: infinitely weak constraint

      const SurveyDet *survey = ali->survey();
      if (survey != NULL) {
         matrix6x6 = survey->errors();  // save the constraint information
      }
      ali->setSurvey(new SurveyDet(ali->surface(), matrix6x6));
   } // end loop over alignables
}

//
// const member functions
//

//
// static member functions
//
