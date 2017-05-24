/*********************************************************
* $Id: RPXMLConfig.h,v 1.1.1.1 2007/05/16 15:44:50 hniewiad Exp $
* $Revision: 1.1.1.1 $
* $Date: 2007/05/16 15:44:50 $
**********************************************************/

#ifndef SimG4CMS_TotemRPProtTranspPar_RPXMLConfig_H
#define SimG4CMS_TotemRPProtTranspPar_RPXMLConfig_H

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMNodeList.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMLSSerializer.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOMImplementationLS.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/framework/LocalFileFormatTarget.hpp>
#include <xercesc/framework/StdOutFormatTarget.hpp>


#include <iostream>
#include <string>
#include <map>
#include <sstream>
#include <vector>


XERCES_CPP_NAMESPACE_USE

void
print (const XMLCh * ch);


// Represents one configuration file with multiple rows on two levels
// for example:
// <config>
//  <item id=.. />
//    <subitem id=.. />
//  <item id=.. />
// </config>
// Provides methods for accessing to file, and getting/setting parameters
class RPXMLConfig
{

public:

// Constructor and destructor
  RPXMLConfig ();
  ~RPXMLConfig ();

// Getters and setters
  void setFilename (std::string filename);
  std::string getFilename ();

// First level access
  std::vector<int> getIds();
  char *get (int id, const char *name);
  template<class F>
    F get (int id, std::string name)
    {
//      std::cout<<this->get(id, name.c_str())<<std::endl;
  //    std::cout<<std::string( this->get(id, name.c_str()) )<<std::endl;
      std::istringstream sin(std::string(this->get(id, name.c_str() )));
      F val;
    //  std::cout<<val<<std::endl;
      sin>>val;
      //std::cout<<val<<std::endl;
      return val;
    }
  int getInt (int id, char *name);
  double getDouble (int id, char *name);
  void set (int id, const char *name, const char *value);
  template<class F>
    void set (int id, std::string name, F value) {std::ostringstream sout; sout<<value; set(id, name.c_str(), sout.str());}
  void setInt (int id, char *name, const int value);
  void setDouble (int id, char *name, const double value);

// Second level access
  std::vector<int> getIds(int id1);
  char *get (int id1, int id2, const char *name);
  template<class F>
    F get (int id1, int id2, std::string name) {std::istringstream sin(std::string(this->get(id1, id2, name.c_str() ))); F val; sin>>val; return val;}
  int getInt (int id1, int id2, char *name);
  double getDouble (int id1, int id2, char *name);
  void set (int id1, int id2, const char *name, const char *value);
  template<class F>
    void set (int id1, int id2, std::string name, F value) {std::ostringstream sout; sout<<value; set(id1, id2, name.c_str(), sout.str());}
  void setInt (int id1, int id2, char *name, const int value);
  void setDouble (int id1, int id2, char *name, const double value);

// Methods for accesing file
  void save ();
  void save (const std::string filename);
  void read ();
  void read (const std::string filename);


private:

  bool isEmpty();
  bool containItem(const int id);
  bool containSubItem(const int id1, const int id2);
  bool containAttr(const int id, const char * name);
  void addItem(const int id);
  void addSubItem(const int id1, const int id2);
  void parse(const std::string filename);

// default filename
  std::string fileName;
// parser
  XercesDOMParser * parser;
// document
 DOMDocument * doc;
// main node
 DOMNode * mn;
// writer
 DOMLSSerializer * writer;

// Constants
 XMLCh * id_string;
 XMLCh * subitem_string;
 XMLCh * item_string;
 XMLCh * config_string;


};


#endif  //SimG4CMS_TotemRPProtTranspPar_RPXMLConfig_H
