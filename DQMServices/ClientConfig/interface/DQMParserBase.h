#ifndef DQMParserBase_H
#define DQMParserBase_H

/** \class DQMParserBase
 * *
 *  Base class for Parsers used by DQM
 *
 *
 *  $Date: 2006/05/04 10:26:52 $
 *  $Revision: 1.1 $
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



using namespace xercesc;

class DQMParserBase{

 public:
	///Creator
	DQMParserBase();
	///Destructor
	~DQMParserBase();
	///Methor that parses the xml file configFile
	void getDocument(std::string configFile);
	///Returns the number of nodes with given name
	int countNodes(std::string tagName);

 protected:	 
	 
	DOMDocument* doc;
	 
	 
 private:	 
	 	 	 


};


#endif
