//-------------------------------------------------
//
/**  \class DTParameterValue
 *  A value for a parameter for Level1 Mu DT Trigger
 *
 *   \author C.Grandi
 */
//
//--------------------------------------------------
#ifndef DT_PARAMETER_VALUE_H_
#define DT_PARAMETER_VALUE_H_

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//----------------------
// Base Class Headers --
//----------------------

//---------------
// C++ Headers --
//---------------
#include <cstdlib>
#include <string>



//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTParameterValue {

 public:
  ///  Constructor
  DTParameterValue();

  ///  Constructor
  DTParameterValue(std::string name, double value);

  ///  Constructor
  DTParameterValue(const DTParameterValue& param);
  
  ///  Destructor 
  ~DTParameterValue();

  /// Assignment operator
  DTParameterValue& operator=(const DTParameterValue& param);

  /// Set the name-value pair
  DTParameterValue& set(std::string name, double value);

  /// Clear
  void clear();

  /// Return the value
  inline double value() const {return value_; }
  
  /// Return the name
  inline std::string name() const { return name_; }
  
 private:
  std::string name_;
  double value_;

};
#endif
