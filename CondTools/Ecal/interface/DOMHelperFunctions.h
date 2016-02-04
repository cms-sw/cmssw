#ifndef _CondTools_Ecal_XMLHelperFunctions_
#define _CondTools_Ecal_XMLHelperFunctions_


/**
 * Helper function for converting Ecal DB Objects to XML
 *
 * \author S. Argiro, F. Rubbo
 *
 * $Id: DOMHelperFunctions.h,v 1.2 2009/10/28 13:34:46 argiro Exp $
 */

#include <xercesc/dom/DOMNode.hpp>
#include "DataFormats/DetId/interface/DetId.h"
#include <string>
#include "CondTools/Ecal/interface/EcalCondHeader.h"
#include "CondTools/Ecal/interface/XMLTags.h"
#include "CondTools/Ecal/interface/XercesString.h"
#include <xercesc/dom/DOM.hpp>
#include <sstream>



namespace xuti {

  /// Assuming \param node is a  <cell> node, read the id 
  const DetId readCellId(xercesc::DOMElement* node);

  /// Append a Cell node with attributes to \param node
  xercesc::DOMElement*  writeCell(xercesc::DOMNode* node, const DetId& detid);
  
  /// get the child of \param node called \param nodedame,return 0 if not found
  xercesc::DOMNode * getChildNode(xercesc::DOMNode * node,  
				  const std::string& nodename );

  /// get the node data as string. Needs to be used to avoid splitting 
  //  white spaces instead of the templatized GetNodeData
  void GetNodeStringData(xercesc::DOMNode* node, std::string& value);
  

  
  /// get the node data
  template <class T> void GetNodeData(xercesc::DOMNode* node, T& value)
    {
      std::string value_s = toNative(node->getTextContent());
      std::stringstream value_ss(value_s);
      value_ss>> value; 
    }
  

  /// write a node with \param tag and \param value under \param parentNode
  template <class T> void WriteNodeWithValue(xercesc::DOMNode* parentNode,
					     const std::string& tag,
					     const T& value)
    {

      xercesc::DOMDocument * doc = parentNode->getOwnerDocument();
      xercesc::DOMElement* new_node = doc->createElement(fromNative(tag).c_str());
      parentNode->appendChild(new_node);
      
      std::stringstream value_ss;
      value_ss <<value;
 
      xercesc::DOMText* tvalue = 
	doc->createTextNode(fromNative(value_ss.str()).c_str());
      new_node->appendChild(tvalue);
      
    }
 

  /// write \param header under \param parentNode
  void writeHeader (xercesc::DOMNode* parentNode, 
		    const EcalCondHeader& header );
  
  /// read header from \param parentNode
  void readHeader(xercesc::DOMNode* parentNode, EcalCondHeader& header); 


  /// read header from any xml file, return -1 in case of error
  int readHeader(const std::string& filename, EcalCondHeader& header);
}


#endif




