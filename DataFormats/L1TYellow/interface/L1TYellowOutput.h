#ifndef L1TYELLOWOUTPUT_H
#define L1TYELLOWOUTPUT_H

#include <stdint.h>
#include <iostream>
#include <vector>

// need to check if DataFormats should be in a namespace...
//namespace l1t {

//  L1TYellowOutput:
//
//  Fictitious output of Yellow Trigger.
//
//  TO DO:
//     - Add Det ID example
//


  class L1TYellowOutput;
  typedef std::vector<L1TYellowOutput> L1TYellowOutputCollection;
  
  class L1TYellowOutput {
  public:
    enum {ET_MASK = 0xff};
    
    /// default constructor
    L1TYellowOutput();
    
    /// destructor
    ~L1TYellowOutput();
    
    /// get raw data
    uint16_t raw() const { return m_data; }
    
    /// get Et
    unsigned et() const { return m_data & ET_MASK; }
    
    /// set data
    void setRawData(uint32_t data) { m_data = data; }
    
    /// print to stream
    friend std::ostream& operator << (std::ostream& os, const L1TYellowOutput& x);
    
  private:
    uint16_t m_data;  
};
  

//} // namespace l1t
#endif /*L1TYELLOWOUT_H*/

