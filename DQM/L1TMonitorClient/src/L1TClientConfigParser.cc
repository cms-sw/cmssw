// $Id$
// $Log$

#include "DQM/L1TMonitorClient/interface/L1TClientConfigParser.h"
#include "DQMServices/ClientConfig/interface/ParserFunctions.h"

#include "boost/algorithm/string/trim.hpp"

using namespace xercesc;

L1TClientConfigParser::L1TClientConfigParser(std::string configFile) :
  DQMParserBase(),
  fname_(configFile)
{
  parse(fname_);
}

bool L1TClientConfigParser::parse(std::string fname)
{
  summaryList_.clear();
  getDocument(fname.c_str());
  
  DOMNodeList * sl  = doc->getElementsByTagName(qtxml::_toDOMS("SummaryList"));
  // loop over tags
  for ( unsigned int i = 0; i < sl->getLength(); ++i ) {
    DOMElement *argNode = dynamic_cast<DOMElement*>(sl->item(i));
    if ( !argNode ) break;
    DOMNodeList *ssl = argNode->getElementsByTagName(qtxml::_toDOMS("Title"));
    MeInfoList a;
    if ( ssl->getLength() == 0 ) {
      // no title for this summary
      a.setTitle("default");
    }
    else {
      DOMElement *n0 = dynamic_cast<DOMElement*>(ssl->item(0));
      if ( n0 ) {
        DOMText *m = dynamic_cast<DOMText*>(n0->getFirstChild());
        if ( m ) {
	  std::string title(qtxml::_toString(m->getData()));
	  
	  boost::trim_if(title, boost::is_any_of(" \n\t"));
          a.setTitle(title);
        }
      }
    }

    // loop over ME's
    ssl = argNode->getElementsByTagName(qtxml::_toDOMS("MonitorElement"));
    for ( unsigned int ii = 0; ii < ssl->getLength(); ++ii ) {
      // MonitorElements have child nodes of type Name and Option
      // the option child is, well, optional.
      DOMElement *e = dynamic_cast<DOMElement*>(ssl->item(ii));
      if ( !e ) continue;
      DOMNodeList *names = e->getElementsByTagName(qtxml::_toDOMS("Name"));
      if ( ! names ) continue; // no name found
      DOMElement *pN = dynamic_cast<DOMElement*>(names->item(0));
      if ( !pN ) continue; // it's not an element
      DOMText *mm = dynamic_cast<DOMText*>(pN->getFirstChild());
      if ( ! mm ) continue;
      std::string meName = qtxml::_toString(mm->getData());
      boost::trim_if(meName, boost::is_any_of(" \n\t"));

      DOMNodeList *options = 
	e->getElementsByTagName(qtxml::_toDOMS("Option"));
      std::string meOption("");
      if ( options && (options->getLength() > 0) ) { // no option found
	DOMElement *pO = dynamic_cast<DOMElement*>(options->item(0));
	if ( !pO ) continue; // it's not an element
	DOMText *m = dynamic_cast<DOMText*>(pO->getFirstChild());
        if ( m ) {
	  meOption = qtxml::_toString(m->getData());
	  boost::trim_if(meOption, boost::is_any_of(" \n\t"));
        }
      }
      a.push_back(MeInfo(meName, meOption));
    }
    summaryList_.push_back(a);
      
      
  }
  return true;
}

L1TClientConfigParser::~L1TClientConfigParser()
{
}

std::ostream & operator << (std::ostream & out,
			    const L1TClientConfigParser::MeInfo &mei )
{
  out << mei.getName() << "(" <<  mei.getOptions() <<")";
  return out;
}

