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
//#include "CondFormats/BTagObjects/interface/CalibrationInterface.h"
//#include "RecoBTag/TrackProbability/interface/CalibrationInterface.h"
#include "RecoBTag/XMLCalibration/interface/CalibrationInterface.h"

namespace XERCES_CPP_NAMESPACE { class DOMNode; }



//template <class T,class CO> class   CalibrationInterface;
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

template <class T,class CO> class AlgorithmCalibration : public CalibrationInterface<T,CO>
{
// friend class CalibrationInterface<T,CO>; 
  public:
    typedef XERCES_CPP_NAMESPACE::DOMElement DOMElement;
    typedef XERCES_CPP_NAMESPACE::DOMNode DOMNode;

  /**
   * Create an AlgorithmCalibration class and load from fileName the
   * cateogries and their objects.
   */
    AlgorithmCalibration(const std::string & fileName);
    
    ~AlgorithmCalibration();
    
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

    CO* readObject(DOMNode *);
    bool readCategories();
    
 protected:

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
    std::string m_filename;
    CalibrationXML *m_xml;
};


template <class T,class CO> 
AlgorithmCalibration<T,CO>::AlgorithmCalibration(const std::string & filename) : m_filename(filename), m_xml(0)
{
 readCategories();
 if(m_xml) {
  m_xml->closeFile();
 }
}

template <class T,class CO> 
AlgorithmCalibration<T,CO>::~AlgorithmCalibration()
{
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
	  CO * obj =readObject(n1->getFirstChild());
          if(obj) 
           {
                this->addEntry(*cat,*obj);  
                delete obj;  
           }
          delete cat;
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

#endif
