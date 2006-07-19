//-------------------------------------------------
//
/**  \class DTParameter
 *    A configurable parameter for Level1 Mu DT Trigger
 *
 *   \author C.Grandi
 *   \author D.Bonacorsi
 *   \author S.Marcellini
 */
//
//--------------------------------------------------
#ifndef DT_PARAMETER_H_
#define DT_PARAMETER_H_

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//----------------------
// Base Class Headers --
//----------------------
#include "L1Trigger/DTUtilities/interface/DTParameterValue.h"

//---------------
// C++ Headers --
//---------------
#include <string>
#include <vector>


//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTParameter {

 public:

  typedef std::vector< DTParameterValue > ValueContainer;
  typedef ValueContainer::iterator          ValueIterator;
  typedef ValueContainer::const_iterator    ValueConstIterator;


  ///  Constructor
  DTParameter();

  ///  Constructor
  DTParameter(std::string label, std::string name);
  ///  Constructor
  DTParameter(const DTParameter& param);  
  ///  Destructor 
  ~DTParameter();
  /// Set the parameter name and label
  DTParameter& setName(std::string label, std::string name) { 
    label_ = label; 
    name_ = name;
    return *this; 
  }

  /// Add a parameter value to the list of valid parameters
  void addValidParam(std::string name, double value);

  /// Add a parameter value to the list of valid parameters
  void addValidParam(const DTParameterValue& val);

  /// Assignment operator
  DTParameter& operator=(const DTParameter& param);

  /// set the current value
  DTParameter& operator=(std::string val);

  /// clear the parameter
  void clear(); 

  /// Return the parameter name
  inline std::string name() const { return name_; }

  /// Return the parameter label
  inline std::string label() const { return label_; }

  /// Return the current value
  inline double currentValue() const { return currentValue_.value(); }
  
  /// Return the current value meaning
  inline std::string currentMeaning() const { return currentValue_.name(); }
  
 private:
  std::string label_;
  std::string name_;
  ValueContainer availableValues_;
  DTParameterValue currentValue_;

};
#endif


