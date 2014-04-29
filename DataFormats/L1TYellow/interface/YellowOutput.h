///
/// \class l1t::YellowOutput
///
/// Description: Output data from the fictitious Yellow trigger.
///
/// Implementation:
///    Demonstrates how to implement a DataFormats class.
///
/// \author: Michael Mulhearn - UC Davis
///

//
//  An example DataFormats class which contains only two fields:  et and yvar.
//
//  Note that bit packing of raw data is discouraged (what if number of bits is
//  adjusted in firmware?)  So separate unsigned integers are used for each.
//
//  To see by example how the DataFormats class is implemented make sure to see:
//
//    DataFormats/L1TYellow/src/YellowOutput.cc  
//    DataFormats/L1TYellow/src/classes.h
//    DataFormats/L1TYellow/src/classes_def.xml
//

#ifndef YELLOWOUTPUT_H
#define YELLOWOUTPUT_H

#include <stdint.h>
#include <iostream>
#include <vector>

namespace l1t {

  class YellowOutput;
  typedef std::vector<YellowOutput> YellowOutputCollection;
  
  class YellowOutput {
  public:
    /// default constructor
    YellowOutput();
    
    /// destructor
    ~YellowOutput();
    
    /// get Et
    unsigned et() const { return m_et; }
    
    // set Et
    void setEt(unsigned et) { m_et = et; }

    /// get Y-variable
    unsigned yvar() const { return m_yvar; }
    
    // set Y-variable
    void setYvar(unsigned yvar) { m_yvar = yvar; }

    /// print to stream
    friend std::ostream& operator << (std::ostream& os, const YellowOutput& x);
    
  private:
    // avoid bit packing, in case number of bits changes:
    unsigned m_et;
    unsigned m_yvar;

  };
  

} // namespace l1t
#endif /*YELLOWOUT_H*/

