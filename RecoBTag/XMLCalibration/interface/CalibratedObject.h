#ifndef CALIBRATED_OBJECT_H
#define CALIBRATED_OBJECT_H
#include <string>
#include <xercesc/dom/DOM.hpp>    

/** CalibratedObject class.
* This is the abstract class of any object that the calibration framework returns to
* the algorithm. 
* Actually in the simplified framework it is not needed to inherit from it, it is 
* enough to mimic the same interface.
* Example of "CalibratedObjects" are PDFs,Neural Network set of parameters, ...
*/

class CalibratedObject 
{
  public:
    /** This function has to be implemented in derived class.
    * It should read all the information the calibrated objects need to
    * load to be initialized from the xml file.
    * It is possible to use CalibrationXML::readAttribute<type>() to read an
    * attribute from the passed DOMElement.
    */
    virtual void read( XERCES_CPP_NAMESPACE::DOMElement * dom ) = 0;

    /** This function has to be implemented in derived class.
    * It should write all the information the calibrated objects need to\
    * save/load.
    * It is possible to use CalibrationXML::writeAttribute() to write an
    * attribute in the passed DOMElement.
    */
    virtual void write( XERCES_CPP_NAMESPACE::DOMElement * dom ) const = 0;

    /** This function has to be implemented in derived class.
    * Prepare the calibrated object for a calibration run.
    * E.g. clear the right data members.
    */  
    virtual void startCalibration() {};

    /** This function has to be implemented in derived class.
    * Calibration is finished. Prepare for writing.
    * E.g. fit histogram, normalize, compute averages, whatever...
    */  
    virtual void finishCalibration() {};

    /** 
    * You have to impelement a different updateCalibration(CalibrationInput) in the derived class for each 
    * CalibrationInput you want to be able to calibrate on.
    * So for example you may want to have an updateCalibration(RecTrack) but also a 
    * updateCalibration(JetWithTracks) that only loops on jet rectracks and calls 
    * updateCalibration(RecTrack).
    *
    * This implementation do nothing. 
    */
    virtual void updateCalibration(){}
    
    /** Return a name for your calibrated object. It is used as XML tag name in reading and writing.
    */
    virtual std::string name() const = 0;     
};

#endif
