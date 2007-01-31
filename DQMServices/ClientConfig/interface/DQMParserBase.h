#ifndef DQMParserBase_H
#define DQMParserBase_H

/** \class DQMParserBase
 * *
 *  Base class for Parsers used by DQM
 *
 *
 *  $Date: 2006/07/25 12:31:29 $
 *  $Revision: 1.3 $
 *  \author Ilaria Segoni
  */

#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMCharacterData.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/util/XMLUni.hpp>
#include <xercesc/util/XMLURL.hpp>

          

#include<iostream>
#include<string>
#include<vector>
#include<map>



class DQMParserBase{

 public:
	///Creator
	DQMParserBase();
	///Destructor
	virtual ~DQMParserBase();
	///Methor that parses the xml file configFile
	void getDocument(std::string configFile);
	///Returns the number of nodes with given name
	int countNodes(std::string tagName);
	///Parses a new Document
	void getNewDocument(std::string configFile);

 protected:	 
	xercesc::XercesDOMParser* parser; 
	xercesc::DOMDocument* doc;
	 
	 
 private:	 
	 	 	 


};


#endif
