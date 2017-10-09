#ifndef CondFormats_DTObjects_DTRecoConditions_H
#define CondFormats_DTObjects_DTRecoConditions_H

/** \class DTRecoConditions
 *  DB object for storing per-SL DT reconstruction parameters (ttrig, vdrift, uncertainties), 
 *  possibly with their dependency from external quantities (like position, angle, etc.) 
 *
 *  Dependencies can be specified  with the expression set by setFormula(string), representing:
 *  -a TFormula,  e.g. "[0]+[1]*x", in the most general case;
 *  -special cases like "[0]" = fixed constant, that are implemented without calling TFormula.
 *
 *  \author N. Amapane, G. Cerminara
 */


#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>
#include <vector>
#include <string>
#include <stdint.h>
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
#include <atomic>
#endif

class DTWireId;
class TFormula;

class DTRecoConditions {
public:
  typedef std::map<uint32_t, std::vector<double> >::const_iterator const_iterator;

  /// Constructor
  DTRecoConditions();
  DTRecoConditions(const DTRecoConditions&);
  const DTRecoConditions& operator=(const DTRecoConditions&);

  /// Destructor
  virtual ~DTRecoConditions();

  void setVersion(int version) {
    theVersion = version;
  }
  
  /// Version numer specifying the structure of the payload. See .cc file for details.
  int version() const {
    return theVersion;
  }

  /// Get the value correspoding to the given WireId, 
  //// using x[] as parameters of the parametrization when relevant
  float get(const DTWireId& wireid, double* x=0) const;

  /// Set the expression representing the formula used for parametrization
  void setFormulaExpr(const std::string& expr) {
    expression=expr;
  }
  
  std::string getFormulaExpr() const {
    return expression;
  }

  /// Fill the payload
  void set(const DTWireId& wireid, const std::vector<double>& values); 

  /// Access the data
  const_iterator begin() const;
  const_iterator end() const;

private:

  // The formula used for parametrization (transient pointer)
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  mutable std::atomic<TFormula*> formula COND_TRANSIENT;
#else
  mutable TFormula* formula COND_TRANSIENT;
#endif
  
  // Formula evalaution strategy, derived from expression and cached for efficiency reasons
#if !defined(__CINT__) && !defined(__MAKECINT__) && !defined(__REFLEX__)
  mutable std::atomic<int> formulaType COND_TRANSIENT;
#else
  mutable int formulaType COND_TRANSIENT;
#endif

  // String with the expression representing the formula used for parametrization
  std::string expression;
  
  // Actual data
  std::map<uint32_t, std::vector<double> > payload;
  
  // Versioning
  int theVersion;

  COND_SERIALIZABLE;
};
#endif

