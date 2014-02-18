///
/// \class l1t::YellowDigi
///
/// Description: Input digis for the fictitious Yellow trigger.
///
/// Implementation:
///    Demonstrates how to implment a DataFormats Class. 
///
/// \author: Michael Mulhearn - UC Davis
///

#ifndef L1TYELLOWDIGI_H
#define L1TYELLOWDIGI_H

//  
//  See YellowOutput.h for more details on implementation.
//

#include <stdint.h>
#include <iostream>
#include <vector>

namespace l1t {

  class YellowDigi;
  typedef std::vector<YellowDigi> YellowDigiCollection;
  
  class YellowDigi {
  public:
    /// default constructor
    YellowDigi();
    
    /// destructor
    ~YellowDigi();
    
    /// get Et
    unsigned et() const { return m_et; }
    
    // set Et
    void setEt(unsigned et) { m_et = et; }

    /// get Y-variable
    unsigned yvar() const { return m_yvar; }
    
    // set Y-variable
    void setYvar(unsigned yvar) { m_yvar = yvar; }

    /// print to stream
    friend std::ostream& operator << (std::ostream& os, const YellowDigi& x);
    
  private:
    unsigned m_et;
    unsigned m_yvar;
  };
  

} // namespace l1t

#endif 

