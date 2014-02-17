#ifndef SiStripConfigWriter_H
#define SiStripConfigWriter_H

/** \class SiStripConfigWriter
 * *
 *  Base class for Parsers used by DQM
 *
 *
 *  $Date: 2008/12/26 09:05:28 $
 *  $Revision: 1.3 $
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
        bool init(std::string main);
        void write(std::string fname);
        void createElement(std::string tag);
        void createElement(std::string tag, std::string name);
        void createChildElement(std::string tag,std::string name);
        void createChildElement(std::string tag,std::string name,std::string att_name,std::string att_val);
        void createChildElement(std::string tag,std::string name,std::string att_name1,std::string att_val1,
                                                                   std::string att_name2,std::string att_val2);
        void createChildElement(std::string tag,std::string name,std::string att_name1,std::string att_val1,
                                                                   std::string att_name2,std::string att_val2,
                                                                   std::string att_name3,std::string att_val3);
        
 protected:	 
	 
	 
	 
 private:
	 
	xercesc::DOMElement* theTopElement;
	xercesc::DOMElement* lastElement;
	xercesc::DOMDocument* theDoc ;
        xercesc::DOMImplementation* domImpl;
        xercesc::DOMWriter* domWriter;

};


#endif
