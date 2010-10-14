#ifndef CALIBRATION_CATEGORY_H
#define CALIBRATION_CATEGORY_H
#include <map>
#include <string>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/util/XMLString.hpp>
#include <stdlib.h>
#include "CalibrationXML.h"



class CalibratedObject ;


/** This class defines a category.
* The class is templated on the CI (category input) 
* To actually use that class you have to inherit from it and implement the pure
* virtual functions.
*/

template <class CI> 
class CalibrationCategory
{
  public:

    typedef XERCES_CPP_NAMESPACE::DOMElement DOMElement;
    typedef  CI Input;

    CalibrationCategory();
    
    virtual ~CalibrationCategory();

/** 
* Should returun true if the CI input match that category 
*/
    virtual bool match(const CI & calibrationInput) const = 0; //Not implemented here
   
    virtual std::string name(){return "BaseCalibrationCategory";} //Set it to pure virtual?
    
  protected:
/** Read category parameters from XML
*/
    virtual void readFromDOM(DOMElement * dom)=0;  //Not implemented
/** Save category parameters to XML
*/
    virtual void saveToDOM(DOMElement * dom)=0;  //Not implemented

    virtual void dump() { }

    
};


template <class CI> CalibrationCategory<CI>::CalibrationCategory() 
{
}

template <class CI>  CalibrationCategory<CI>::~CalibrationCategory()
{
}




#endif
