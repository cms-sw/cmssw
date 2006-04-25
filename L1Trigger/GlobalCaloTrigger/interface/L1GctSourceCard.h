#ifndef L1GCTSOURCECARD_H_
#define L1GCTSOURCECARD_H_

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctRegion.h"

#include <vector>
#include <bitset>
#include <fstream>
#include <string>
#include <iostream>

typedef unsigned long int ULong;
typedef unsigned short int UShort;

class L1RCTCrate;

/*
 * \author Jim Brooke & Robert Frazier
 * \date April 2006
 */

/*! \class L1GctSourceCard
 * \brief Represents a GCT Source Card
 *
 *  Can be constructed to be one of three different variants of 
 *  source card, depending on which pairs of RCT crate output
 *  cables are being read in.
 */

/**
  * 
  *RCT Input File Format 
  *Line 1: Crossing no as "Crossing x" (2)     
  *Line 2: isoe0 isoe1 isoe2 isoe3 nonIsoe0 nonIsoe1 nonIso2 nonIso3 (8) 
  *Line 3: RC0mip0 RC0mip1 RC1mip0 RC1mip1 RC2mip0 RC2mip1 RC3mip0 RC3mip1 RC4mip0 RC4mip1 
  *        RC5mip0 RC5mip1 RC6mip0 RC6mip1 (14)
  *Line 4: RC0qt0 RCqt1 RC1qt0 RC1qt1 RC2qt0 RC2qt1 RC3qt0 RC3qt1 RC4qt0 RC4qt1 
  *        RC5qt0 RC5qt1 RC6qt0 RC6qt1 (14)
  *Line 5: RC0reg0 RC0reg1 RC1reg0 RC1reg1 RC2reg0 RC2reg1 RC3reg0 RC3reg1 RC4reg0 RC4reg1
  *        RC5reg0 RC5reg1 RC6reg0 RC6reg1 (14)
  *Line 6: HF0eta0 HF0eta1 HF0eta2 HF0eta3 HF1eta0 HF1eta1 HF1eta2 HF1eta3 (8)
  *...
  *... 

NOTE:  CMS IN 2004/009 specifies that cable four provides 8 Quiet bits for the HF.  These are not
       detailed in the fileformat above, and are not currently dealt with in any way.
  */ 


/*TO DO: 1) Additional (file) error handling maybe? Currently done with some debug asserts.
 *       2) Currently doesn't like any whitespace after final entry in input file.
 *       3) Need to sort out BuildFiles so the CMS Exception class can be used
*/

class L1GctSourceCard
{
public:
    /// cardType1 reads cables 1&2, cardType2 reads cables 3&4, cardType3 reads cables 5&6
    enum SourceCardType{cardType1 =1, cardType2, cardType3};
    static const int MIP_BITWIDTH = 14;
    static const int QUIET_BITWIDTH = 14;
    typedef std::bitset<MIP_BITWIDTH> MipBits;
    typedef std::bitset<QUIET_BITWIDTH> QuietBits;

    /// typeVal determines which pairs of cables to read, according to the SourceCardType enumeration
    L1GctSourceCard(SourceCardType typeVal, L1RCTCrate* rc=0);
    ~L1GctSourceCard();
  
    /// Open input file
    void openInputFile(std::string fileName);
  
    /// Read next event and push data into the relevant buffers
    void readBX();

    /// return true if current event is valid (false if EOF reached!)
    bool dataValid() const { return !m_fin.eof(); }

    /// close input file
    void closeInputFile() { m_fin.close(); }

    /// clear the buffers
    void reset();
    
    /// get the data from RCT/File/event/...
    void fetchInput();

    /// process the event
    void process();
  
    //Methods to read the BX data out from the class
    std::vector<L1GctEmCand> getIsoElectrons() const;
    std::vector<L1GctEmCand> getNonIsoElectrons() const;
    MipBits getMipBits() const;
    QuietBits getQuietBits() const;
    std::vector<L1GctRegion> getRegions() const;
        
    /// Returns the value of the current bunch crossing number
    long int getBxNum() const { return m_currentBX; }

private:

    //SYMBOLIC CONSTANTS
    static const int NUM_ELEC = 4;
    static const int NUM_REG_TYPE2 = 12;
    static const int NUM_REG_TYPE3 = 10;
    static const int DATA_OFFSET_TYPE3 = NUM_ELEC*2 + MIP_BITWIDTH + QUIET_BITWIDTH;
    static const int DATA_OFFSET_TYPE2 = NUM_ELEC*2 + MIP_BITWIDTH + QUIET_BITWIDTH + NUM_REG_TYPE3;    

    //PRIVATE MEMBER VARIABLES
    /// pointer to the RCT crate
    L1RCTCrate* m_rctCrate;
    
    /// SourceCard type
    SourceCardType m_cardType;

    /// file handle
    std::ifstream m_fin;

    /// Stores the current bunch crossing number
    long int m_currentBX;

    //Data buffers
    std::vector<L1GctEmCand> m_isoElectrons;  
    std::vector<L1GctEmCand> m_nonIsoElectrons;
    MipBits m_mipBits;
    QuietBits m_quietBits;
    std::vector<L1GctRegion> m_regions;
    
    //PRIVATE MEMBER FUNCTIONS
    ///Sets the appropriate sizes of vector depending on type of source card 
    void setVectorSizes();
    
    void getCables1And2();  ///< Reads in data corresponding to RCT crate cables 1 & 2
    void getCables3And4();  ///< Reads in data corresponding to RCT crate cables 3 & 4
    void getCables5And6();  ///< Reads in data corresponding to RCT crate cables 5 & 6

    /// Reads the Bunch Crossing number from the file
    void readBxNum();   
    
    ///Changes an RCT output ULong into an EmCand with 6bits of rank, the 'region ID' stored in phi, and 'card ID' stored in eta
    L1GctEmCand convertToEmCand(ULong& rawData) const;
    
    ///Changes an RCT output ULong into a Region with 10bits Et, overflow, and tauVeto.
    L1GctRegion convertToCentralRegion(ULong& rawData) const;
};

#endif /*L1GCTSOURCECARD_H_*/
