#ifndef DATAFORMATS_HCALDIGI_HCALTTPDIGI_H
#define DATAFORMATS_HCALDIGI_HCALTTPDIGI_H 1

#include <stdint.h>
#include <vector>
#include <ostream>

/** \class HcalTTPDigi
  *  
  * $Date: 2010/04/07 02:06:12 $
  * $Revision: 1.2 $
  * \author J. Mans - Minnesota
  */
class HcalTTPDigi {
public:
    typedef int key_type; // Needed for the sorted collection
    HcalTTPDigi();
    HcalTTPDigi(int identifier, int samples, int presamples, unsigned int fwVersion, int algorithm, unsigned int lPipe);
    
    void setSample(int relativeSample,const uint16_t* triggerInputs, const uint32_t algodep, const uint8_t outputTrigger);

    /** get the input bit pattern for the given sample (relative to the SOI)
        the vector will be empty if there is no data for the requested sample
    */
    std::vector<bool> inputPattern(int relativeSample=0) const;
    /** get the output trigger bit set for the given sample (relative to the SOI)
        the vector will be empty if there is no data for the requested sample
    */
    uint8_t triggerOutput(int relativeSample=0) const;
    /** get the "algorithm-dependent-word" for the given sample */
    uint32_t algorithmWord(int relativeSample=0) const;
    
    int id() const { return identifier_ ; } 
    int size() const { return samples_ ; } 
    int presamples() const { return presamples_ ; } 
    int algorithm() const { return algorithm_ ; }
    unsigned int fwVersion() const { return fwVersion_ ; }
    unsigned int pipelineLength() const { return lPipe_ ; } 

    bool operator==(const HcalTTPDigi& digi) const ;
    bool operator!=(const HcalTTPDigi& digi) const { return !(*this == digi) ; } 

private:
    int identifier_;
    int samples_, presamples_; 
    unsigned int fwVersion_ ;
    int algorithm_ ;
    unsigned int lPipe_ ;
    
    uint16_t triggerInputs_[5*8]; // length = 5*samples_
    uint32_t algoDepend_[8]; // length = samples_
    uint8_t triggerOutput_[8]; // length = samples_
};

std::ostream& operator<<(std::ostream&, const HcalTTPDigi&);

#endif
