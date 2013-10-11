#ifndef YELLOWOUTPUT_H
#define YELLOWOUTPUT_H

#include <stdint.h>
#include <iostream>
#include <vector>

//  YellowOutput:
//
//  Fictitious output of Yellow Trigger.
//
//  TO DO:
//     - Add Det ID example
//

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

