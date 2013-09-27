#ifndef L1TYELLOWDIGI_H
#define L1TYELLOWDIGI_H

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


  class L1TYellowDigi;
  typedef std::vector<L1TYellowDigi> L1TYellowDigiCollection;
  
  class L1TYellowDigi {
  public:
    enum {ET_MASK = 0xff};
    
    /// default constructor
    L1TYellowDigi();
    
    /// destructor
    ~L1TYellowDigi();
    
    /// get raw data
    uint16_t raw() const { return m_data; }
    
    /// get Et
    unsigned et() const { return m_data & ET_MASK; }
    
    /// set data
    void setRawData(uint32_t data) { m_data = data; }
    
    /// print to stream
    friend std::ostream& operator << (std::ostream& os, const L1TYellowDigi& x);
    
  private:
    uint16_t m_data;  
};
  

//} // namespace l1t
#endif /*L1TYELLOWOUT_H*/

