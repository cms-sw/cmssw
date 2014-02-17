#ifndef SiPixelConfigWriter_H
#define SiPixelConfigWriter_H

/** \class SiPixelConfigWriter
 * *
 *  Base class for Parsers used by DQM
 *
 *
 *  $Date: 2007/02/20 21:29:25 $
 *  $Revision: 1.2 $
 *  \author Petra Merkel
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




class SiPixelConfigWriter{

 public:
	///Creator
	SiPixelConfigWriter();
	///Destructor
	~SiPixelConfigWriter();
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
