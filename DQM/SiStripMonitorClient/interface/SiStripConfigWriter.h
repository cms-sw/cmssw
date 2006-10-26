#ifndef SiStripConfigWriter_H
#define SiStripConfigWriter_H

/** \class SiStripConfigWriter
 * *
 *  Base class for Parsers used by DQM
 *
 *
 *  $Date: 2006/08/01 18:14:27 $
 *  $Revision: 1.1 $
 *  \author Suchandra Dutta
 */
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/framework/StdOutFormatTarget.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>

#include <xercesc/util/XMLString.hpp>
#include <xercesc/dom/DOMException.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOM.hpp>


          
#include<iostream>
#include<string>
#include<vector>
#include<map>




class SiStripConfigWriter{

 public:
	///Creator
	SiStripConfigWriter();
	///Destructor
	~SiStripConfigWriter();
	///Write XML file
        bool init();
        void write(std::string& fname);
        void createLayout(std::string& name);
        void createRow();
        void createColumn(std::string& element, std::string& name);
        
 protected:	 
	 
	 
	 
 private:
	 
	xercesc::DOMElement* theTopElement;
	xercesc::DOMElement* lastLayout;
	xercesc::DOMElement* lastRow;
	xercesc::DOMDocument* theDoc ;
        xercesc::DOMImplementation* domImpl;
        xercesc::DOMWriter* domWriter;

};


#endif
