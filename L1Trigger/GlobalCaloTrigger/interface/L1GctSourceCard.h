#ifndef L1GCTSOURCECARD_H_
#define L1GCTSOURCECARD_H_


#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctDigis.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctRegion.h"

#include <vector>
#include <bitset>
#include <fstream>
#include <string>
#include <iostream>

typedef unsigned long int ULong;
typedef unsigned short int UShort;

/*
 * \class L1GctSourceCard
 * \brief Represents a GCT Source Card
 * \author Jim Brooke & Robert Frazier
 * \date April 2006
 *
 * Can be constructed to be one of three different variants of
 *  source card, depending on which pairs of RCT crate output
 *  cables are being read in.
 */


/**
  * 
  *Rct Input File Format 
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

  /// typeVal determines which pairs of cables to read, according to the SourceCardType enumeration
  L1GctSourceCard(int id, SourceCardType typeVal);
  ~L1GctSourceCard();

  /// Overload << operator
  friend std::ostream& operator << (std::ostream& os, const L1GctSourceCard& card);

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

  // methods to set data in the card for use when pushing from RCT

  /// set the Regions
  void setRegions(std::vector<L1GctRegion> regions);

  /// set the Iso Em candidates
  void setIsoEm(std::vector<L1GctEmCand> isoEm);

  /// set the Non Iso Em candidates
  void setNonIsoEm(std::vector<L1GctEmCand> nonIsoEm);

  /// set the Mip bits
  void setMipBits(unsigned mip);

  /// set the Quiet bits
  void setQuietBits(unsigned quiet);

  //Methods to read the BX data out from the class
  std::vector<L1GctEmCand> getIsoElectrons() const;
  std::vector<L1GctEmCand> getNonIsoElectrons() const;
  unsigned getMipBits() const;
  unsigned getQuietBits() const;
  std::vector<L1GctRegion> getRegions() const;
      
  /// Returns the value of the current bunch crossing number
  long int getBxNum() const { return m_currentBX; }

 private:  // methods

  //PRIVATE MEMBER FUNCTIONS
  ///Sets the appropriate sizes of vector depending on type of source card 
  void setVectorSizes();
  
  void getCables1And2();  ///< Reads in data corresponding to RCT crate cables 1 & 2
  void getCables3And4();  ///< Reads in data corresponding to RCT crate cables 3 & 4
  void getCables5And6();  ///< Reads in data corresponding to RCT crate cables 5 & 6

  /// Reads the Bunch Crossing number from the file
  void readBxNum();  

  L1GctRegion makeRegion(ULong rctFileData);
  L1GctEmCand makeEmCand(ULong rctFileData, bool iso);


 private:  // members

  static const int NUM_ELEC;
  static const int NUM_REG_TYPE2; ///< No. regions type 2 card takes in
  static const int NUM_REG_TYPE3; ///< No. regions type 3 card takes in
  static const int N_MIP_BITS;  // number of MIP bits in file format
  static const int N_QUIET_BITS;  // number of quiet bits in file format
  static const int DATA_OFFSET_TYPE3; ///< Data offset for file reading
  static const int DATA_OFFSET_TYPE2; ///< Data offset for file reading  

  /// card ID
  int m_id;
  ///
  /// SourceCard type
  SourceCardType m_cardType;

  /// file handle
  std::ifstream m_fin;

  /// Stores the current bunch crossing number
  long int m_currentBX;

  //Data buffers
  std::vector<L1GctEmCand> m_isoElectrons;  
  std::vector<L1GctEmCand> m_nonIsoElectrons;
  unsigned m_mipBits;   //IMPORTANT - the static_casts in the getCables1And2 method should
  unsigned m_quietBits; //match the types of m_mipBits and m_quietBits.
  std::vector<L1GctRegion> m_regions;
  
    
};

std::ostream& operator << (std::ostream& os, const L1GctSourceCard& card);

#endif /*L1GCTSOURCECARD_H_*/
