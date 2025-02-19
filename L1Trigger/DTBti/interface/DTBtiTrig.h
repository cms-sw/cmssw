//-------------------------------------------------
//
/**  \class DTBtiTrig
 *
 *    BTI Trigger Data
 *    Has pointers to parent BTI and DTDigis
 *    which allow algorithm debugging
 *
 *
 *   $Date: 2006/07/19 10:18:31 $
 *   $Revision: 1.1 $
 *
 *   \author  C. Grandi
 *
 */
//
//--------------------------------------------------
#ifndef DT_BTI_TRIG_H
#define DT_BTI_TRIG_H

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
class DTDigi;
class DTBtiChip;

//----------------------
// Base Class Headers --
//----------------------
#include "L1Trigger/DTUtilities/interface/DTTrigData.h"
#include "L1Trigger/DTBti/interface/DTBtiTrigData.h"


//---------------
// C++ Headers --
//---------------
#include <vector>

//              ---------------------
//              -- Class Interface --
//              ---------------------

class DTBtiTrig : public DTTrigData {

  public:

  //! Constructor
  DTBtiTrig();

  //! Constructor
  DTBtiTrig(DTBtiChip*, int);

  //! Constructor
  DTBtiTrig(DTBtiChip* tparent, int, int, int, int, int);
 
  //! Constructor
  DTBtiTrig(DTBtiChip* tparent, int, int, int, int, int, int, float*);

  //! Constructor
  DTBtiTrig(DTBtiChip*, DTBtiTrigData);
  
  //! Destructor 
  ~DTBtiTrig();

  //! Set the parent DTBtiChip
  inline void setParent(DTBtiChip* parent) { 
    _tparent = parent; 
  }  

  //! Add a digi to the list
  inline void addDigi(const DTDigi* digi) { 
    _digi.push_back(digi); 
  }

  //! Set trigger step
  inline void setStep(int step) {
    _data.setStep(step);
  }

  //! Set trigger code
  inline void setCode(int code) {
    _data.setCode(code);
  }

  //! Set trigger K parameter
  inline void setK(int k) {
    _data.setK(k);
  }

  //! Set trigger X parameter
  inline void setX(int x) {
    _data.setX(x);
  }

  //! Set triggering equation
  inline void setEq(int eq) {
    _data.setEq(eq);
  }

  //! Clear
  inline void clear() {
    _data.clear();
    _digi.clear();
  }    

  // Const methods

  //! Return chamber identifier
  inline DTChamberId ChamberId() const {
    return _data.ChamberId(); 
  }

  //! Print
  inline void print() const { 
    _data.print();
  }

  //! Return parent BTI pointer
  inline DTBtiChip* tParent() const { 
    return _tparent; 
  }
  
  //! Return the data part
  inline DTBtiTrigData data() const {
    return _data;
  }

  //! Return parent BTI number
  inline int btiNumber() const { 
    return _data.btiNumber(); 
  }
  
  //! Return parent BTI superlayer
  inline int btiSL() const { 
    return _data.btiSL(); 
  }
  
  //! Return trigger step
  inline int step() const { 
    return _data.step(); 
  }

  //! Return trigger code
  inline int code() const { 
    return _data.code(); 
  }
  
  //! Return trigger K parameter
  inline int K() const { 
    return _data.K(); 
  }
  
  //! Return trigger X parameter
  inline int X() const { 
    return _data.X(); 
  }

  //! Return triggering equation
  inline int eq() const { 
    return _data.eq(); 
  }

  //! Return the digi list
  std::vector<const DTDigi*> digiList() const { 
    return _digi; 
  }

  private:

  // Parent BTI
  DTBtiChip* _tparent;

  // Trigger data component
  DTBtiTrigData _data;

  // vector of digi in the cells of the DTBtiChip trigger
  std::vector<const DTDigi*> _digi;

};
#endif
