#ifndef ALGORITHM_CALIBRATION_H
#define ALGORITHM_CALIBRATION_H

#include <map>
#include <string>
#include <vector>
#include <xercesc/dom/DOM.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/util/XMLString.hpp>
#include <string>
#include <list>
#include <iostream>

#include "CalibrationXML.h"

class XERCES_CPP_NAMESPACE::DOMNode;

using namespace std;
using namespace XERCES_CPP_NAMESPACE;

/**
* The AlgorithmCalibration class is the interface between user code and calibration framework 
* developed for BReco subsytem.
* Both the analysis programs and the calibration programs should only access calibration information
* via that class.
*
* An algorithm is supposed to keep a single instance of that class to avoid the reload of
* XML for every  event (during analysis) and because you want to accumulate over all events the data 
* to perform a new calibration in the calibration program.
* 
* The class is templated on the Category T and on the CalibratedObject CO
*/

template <class T,class CO> class AlgorithmCalibration
{
  public:

  /**
   * Create an AlgorithmCalibration class and load from fileName the
   * cateogries and their objects.
   */
    AlgorithmCalibration(const std::string & fileName);
    
    ~AlgorithmCalibration();
    
   /**
    * Get the calibratedObject that belong to the category that matches the given input
    */ 
    const CO* fetch(const typename T::Input & calibrationInput) const ;

    /**
     * Prepare for a new calibration run
     */
    void startCalibration();

    /**
     * Accumulate information to calibrate.
     */
    void updateCalibration(const typename T::Input & calibrationInput);
    template <class CI> void updateCalibration(const typename T::Input & calibrationInputForCategory,const CI & inputForCalibration);

    /**
     * Finalize and save the calibration run to fileName
     */
    void saveCalibration(const std::string & fileName);

 protected:

    pair<T *,CO*> searchCategory(const typename T::Input & calibrationInput) const ;

    CO* readObject(DOMNode *);

    bool readCategories();
    
 protected:

    void addCategory(T * newCategory,CO * newCalibratedObject)
    {
       m_categoriesWithObjects.push_back(pair<T*,CO*>(newCategory,newCalibratedObject));
    }


    DOMElement * dom() 
    {
      if(m_xml == 0)
       {
        m_xml=new CalibrationXML();
	m_xml->openFile(m_filename);
	
       } 
    return m_xml->calibrationDOM();
    }

  private:
    vector<pair<T*,CO*> > m_categoriesWithObjects;
    string m_filename;
    CalibrationXML *m_xml;
};


template <class T,class CO> 
AlgorithmCalibration<T,CO>::AlgorithmCalibration(const string & filename) : m_filename(filename), m_xml(0)
{
 readCategories();
}

template <class T,class CO> 
AlgorithmCalibration<T,CO>::~AlgorithmCalibration()
{
 std::cout << "Algorithm calibration destructor" <<endl;
   for(typename vector<pair<T*,CO*> >::iterator it = m_categoriesWithObjects.begin();it!=m_categoriesWithObjects.end();it++)
   {
    delete (*it).first;
    delete (*it).second;
   }
 
   if(m_xml) delete m_xml;
    
}
    
template <class T,class CO> 
bool AlgorithmCalibration<T,CO>::readCategories()
{
   if(dom()==0) return false;
   
   DOMNode* n1 = dom()->getFirstChild();
   while(n1)
    {
      if (n1->getNodeType() == DOMNode::ELEMENT_NODE   )
      {
	  T *cat = new T();
	  cat->readFromDOM((DOMElement *)n1);
	  addCategory(cat,readObject(n1->getFirstChild()));  

      }
      n1 = n1->getNextSibling();
    }
    
 return true;
}
template <class T,class CO>
CO * AlgorithmCalibration<T,CO>::readObject(DOMNode * dom)
{
  DOMNode* n1 = dom;
  while(n1)
    {
      if (n1->getNodeType() == DOMNode::ELEMENT_NODE   )
	   break;
      n1 = n1->getNextSibling();
    }
     
  if(n1==0) return 0; //Cannot find any calibrated objects
  
  CO * co = new CO();
  co->read((DOMElement *)n1);
  return co;
}

template <class T,class CO> 
pair<T *,CO*>  AlgorithmCalibration<T,CO>::searchCategory(const typename T::Input & calibrationInput) const
{
   pair<T *,CO*> categoryWithObject(0,0);
   for(typename vector<pair<T *,CO *> >::const_iterator it = m_categoriesWithObjects.begin();it!=m_categoriesWithObjects.end();it++)
   {
    
      if((*it).first->match(calibrationInput))
      {
        if(categoryWithObject.first!=0) std::cout << "WARNING: OVERLAP in categories, using latest one" << endl;
        categoryWithObject=*it;
      }
   }
  return categoryWithObject;
}

template <class T,class CO> 
const CO * AlgorithmCalibration<T,CO>::fetch(const typename T::Input & calibrationInput) const
{
  return searchCategory(calibrationInput).second; 
}

template <class T,class CO> 
void AlgorithmCalibration<T,CO>::startCalibration()
{
   for(typename vector<pair<T*,CO*> >::iterator it = m_categoriesWithObjects.begin();it!=m_categoriesWithObjects.end();it++)
   {
    if((*it).second==0)  (*it).second=new CO();
    (*it).second->startCalibration();
   }
}


template <class T,class CO> 
template <class CI> 
void AlgorithmCalibration<T,CO>::updateCalibration(const typename T::Input & calibrationInputForCategory,
						const CI & inputForCalibration)
{
  pair<T *,CO*>categoryWithObject= searchCategory(calibrationInputForCategory); 
  if(categoryWithObject.first==0) std::cout << "No category found for this input" << endl;
   else
  categoryWithObject.second->updateCalibration(inputForCalibration);
}

template <class T,class CO> 
void AlgorithmCalibration<T,CO>::updateCalibration(const typename T::Input & calibrationInput)
{
  updateCalibration(calibrationInput,calibrationInput);
}


template <class T,class CO> 
void AlgorithmCalibration<T,CO>::saveCalibration(const std::string & fileName)
{      
   DOMNode* n1 = dom()->getFirstChild();
  while(n1)   
  { 
    dom()->removeChild(n1);
    n1 = dom()->getFirstChild();
  }
   n1 = dom();	
   for(typename vector<pair<T*,CO*> >::iterator it = m_categoriesWithObjects.begin();it!=m_categoriesWithObjects.end();it++) 
     {
       DOMElement * categoryDom = CalibrationXML::addChild((DOMElement *)n1,(*it).first->name());	
       
       (*it).first->saveToDOM(categoryDom);
       if((*it).second)
       {
        
          DOMElement * objectDom = CalibrationXML::addChild(categoryDom,(*it).second->name());	
           (*it).second->finishCalibration();
           (*it).second->write(objectDom);
       }
       
     }
  if(m_xml) m_xml->saveFile(fileName); 
}



#endif
