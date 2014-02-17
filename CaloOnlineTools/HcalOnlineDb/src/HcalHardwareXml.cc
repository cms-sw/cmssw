// -*- C++ -*-
//
// Package:     XMLTools
// Class  :     HcalHardwareXml
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Gena Kukartsev, kukarzev@fnal.gov
//         Created:  Tue Feb 25 14:30:20 CDT 2008
// $Id: HcalHardwareXml.cc,v 1.6 2010/08/06 20:24:13 wmtan Exp $

#include <sstream>

#include "CaloOnlineTools/HcalOnlineDb/interface/HcalHardwareXml.h"
#include "CalibCalorimetry/HcalTPGAlgos/interface/XMLProcessor.h"

HcalHardwareXml::HcalHardwareXml() : XMLDOMBlock()
{

  DOMElement* rootElem = document -> getDocumentElement();

  partsElem = document->createElement( XMLString::transcode("PARTS") );
  rootElem->appendChild(partsElem);
}


HcalHardwareXml::HcalHardwareXml( std::string _type ) : XMLDOMBlock()
{

  DOMElement* rootElem = document -> getDocumentElement();

  partsElem = document->createElement( XMLString::transcode("PARTS") );
  rootElem->appendChild(partsElem);
}




int HcalHardwareXml::addHardware( std::map<std::string,std::map<std::string,std::map<std::string,std::map<int,std::string> > > > & hw_map )
{
  //hw_map["HO001"]["HORBX09"]["200351"][2]="101295";

  std::map<std::string,int> double_entry; // to check for double entries

  std::map<std::string,std::map<std::string,std::map<std::string,std::map<int,std::string> > > >::const_iterator rbx_slot;
  for (rbx_slot = hw_map . begin(); rbx_slot != hw_map . end(); rbx_slot++){

    HcalPart _p;
    _p . mode = "find";
    _p . kind_of_part = "HCAL RBX Slot";
    _p . name_label = rbx_slot -> first;
    DOMElement * rbx_slot_elem = addPart( partsElem, _p );
    DOMElement * rbx_slot_children_elem = (DOMElement *)rbx_slot_elem -> getElementsByTagName(XMLString::transcode("CHILDREN"))->item(0);

    std::map<std::string,std::map<std::string,std::map<int,std::string> > >::const_iterator rbx;
    for (rbx = rbx_slot->second . begin(); rbx != rbx_slot->second . end(); rbx++){
      HcalPart _p2;
      _p2 . mode = "find";
      _p2 . kind_of_part = "HCAL RBX";
      _p2 . name_label = rbx -> first;
      DOMElement * rbx_elem = addPart( rbx_slot_children_elem, _p2 );
      DOMElement * rbx_children_elem = (DOMElement *)rbx_elem -> getElementsByTagName(XMLString::transcode("CHILDREN"))->item(0);

      std::map<std::string,std::map<int,std::string> >::const_iterator rm;
      for (rm = rbx->second . begin(); rm != rbx->second . end(); rm++){
	HcalPart _p3;
	_p3 . mode = "find";
	_p3 . kind_of_part = "HCAL Readout Module";
	_p3 . barcode = rm -> first;
	DOMElement * rm_elem = addPart( rbx_children_elem, _p3 );
	DOMElement * rm_children_elem = (DOMElement *)rm_elem -> getElementsByTagName(XMLString::transcode("CHILDREN"))->item(0);
	
	std::map<int,std::string>::const_iterator qie;
	for (qie = rm->second . begin(); qie != rm->second . end(); qie++){
	  HcalPart _p4;
	  _p4 . mode = "find";
	  _p4 . kind_of_part = "HCAL QIE Card";
	  _p4 . barcode = qie -> second;
	  _p4 . comment = "HCAL hardware remapping by Gena Kukartsev";
	  _p4 . attr_name = "QIE Card Position";
	  std::stringstream _buffer;
	  _buffer . str("");
	  _buffer << qie->first;
	  _p4 . attr_value = _buffer . str();

	  // check for multiple QIE entries
	  unsigned int _nqie = double_entry.size();
	  double_entry[_p4 . barcode]++;
	  if (double_entry.size() == _nqie){
	    std::cout << "QIE #" << _p4.barcode << " found " << double_entry[_p4 . barcode] << "times!" << std::endl;
	  }else{
	    addPart( rm_children_elem, _p4 );
	  }
	}      
      }      
    }
  }

  return 0;
}

DOMElement * HcalHardwareXml::addPart( DOMElement * parent, HcalPart & part )
{
  
  DOMElement * child    = document -> createElement( XMLString::transcode( "PART" ) );
  child -> setAttribute( XMLString::transcode("mode"), XMLString::transcode(part.mode.c_str()) );

  DOMElement * kindElem = document -> createElement( XMLString::transcode( "KIND_OF_PART" ) );
  DOMElement * nameElem = document -> createElement( XMLString::transcode( "NAME_LABEL" ) );
  DOMElement * barElem  = document -> createElement( XMLString::transcode( "BARCODE" ) );
  DOMElement * childrenElem  = document -> createElement( XMLString::transcode( "CHILDREN" ) );
  DOMElement * commentElem  = document -> createElement( XMLString::transcode( "COMMENT_DESCRIPTION" ) );

  child -> appendChild( kindElem );
  DOMText * _kind = document -> createTextNode( XMLString::transcode(part.kind_of_part.c_str()));
  kindElem -> appendChild( _kind );
  if (part.name_label . size() > 0){
    child -> appendChild( nameElem );
    DOMText * _name = document -> createTextNode( XMLString::transcode(part.name_label.c_str()));
    nameElem -> appendChild( _name );
  }
  if (part.barcode . size() > 0){
    child -> appendChild( barElem );
    DOMText * _code = document -> createTextNode( XMLString::transcode(part.barcode.c_str()));
    barElem -> appendChild( _code );
  }
  if (part.comment . size() > 0){
    child -> appendChild( commentElem );
    DOMText * _comment = document -> createTextNode( XMLString::transcode(part.comment.c_str()));
    commentElem -> appendChild( _comment );
  }
  if (part.attr_name . size() > 0 || part.attr_value . size() > 0){
    DOMElement * predefined_attr_elem  = document -> createElement( XMLString::transcode( "PREDEFINED_ATTRIBUTES" ) );
    child -> appendChild( predefined_attr_elem );
    DOMElement * attr_elem  = document -> createElement( XMLString::transcode( "ATTRIBUTE" ) );
    predefined_attr_elem -> appendChild( attr_elem );
    DOMElement * attr_name_elem  = document -> createElement( XMLString::transcode( "NAME" ) );
    attr_elem -> appendChild( attr_name_elem );
    DOMText * _name = document -> createTextNode( XMLString::transcode(part.attr_name.c_str()));
    attr_name_elem -> appendChild( _name );
    DOMElement * attr_value_elem  = document -> createElement( XMLString::transcode( "VALUE" ) );
    attr_elem -> appendChild( attr_value_elem );
    DOMText * _value = document -> createTextNode( XMLString::transcode(part.attr_value.c_str()));
    attr_value_elem -> appendChild( _value );
  }
  child -> appendChild( childrenElem );

  parent -> appendChild( child );

  return child;
}



HcalHardwareXml::~HcalHardwareXml()
{
}


