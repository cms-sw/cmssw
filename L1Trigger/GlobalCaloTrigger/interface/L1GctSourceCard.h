#ifndef L1GCTSOURCECARD_H_
#define L1GCTSOURCECARD_H_

#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"
#include "DataFormats/L1CaloTrigger/interface/L1CaloRegion.h"

#include <vector>
#include <fstream>
#include <string>
#include <iostream>

// Original Author:  Jim Brooke, Rob Frazier
//         Created:  April, 2006
// $Id: 


///
/// \class L1GctSourceCard
/// \brief Represents a GCT Source Card
///
/// Can be constructed to be one of three different variants of
///  source card, depending on which pairs of RCT crate output
/// cables are being read in.
///
///

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

  /// clear the buffers
  void reset();
  
  /// process the event
  void process();

  // methods to set data in the card for use when pushing from RCT

  /// set the Regions
  void setRegions(std::vector<L1CaloRegion> regions);

  /// set the Iso Em candidates
  void setIsoEm(std::vector<L1CaloEmCand> isoEm);

  /// set the Non Iso Em candidates
  void setNonIsoEm(std::vector<L1CaloEmCand> nonIsoEm);

  /// set the Mip bits
  void setMipBits(unsigned mip);

  /// set the Quiet bits
  void setQuietBits(unsigned quiet);

  //Methods to read the BX data out from the class
  std::vector<L1CaloEmCand> getIsoElectrons() const;
  std::vector<L1CaloEmCand> getNonIsoElectrons() const;
  unsigned getMipBits() const;
  unsigned getQuietBits() const;
  std::vector<L1CaloRegion> getRegions() const;
      
  /// Returns the value of the current bunch crossing number
  long int getBxNum() const { return m_currentBX; }

 private:  // methods

  ///Sets the appropriate sizes of vector depending on type of source card 
  void setVectorSizes();
  
 private:  // members

  static const int NUM_ELEC;
  static const int NUM_REG_TYPE2; ///< No. regions type 2 card takes in
  static const int NUM_REG_TYPE3; ///< No. regions type 3 card takes in
  static const int N_MIP_BITS;  // number of MIP bits in file format
  static const int N_QUIET_BITS;  // number of quiet bits in file format
  static const int DATA_OFFSET_TYPE3; ///< Data offset for file reading
  static const int DATA_OFFSET_TYPE2; ///< Data offset for file reading  

  /// card ID
  unsigned m_id;
  ///
  /// SourceCard type
  SourceCardType m_cardType;

  /// Stores the current bunch crossing number
  long int m_currentBX;

  //Data buffers
  std::vector<L1CaloEmCand> m_isoElectrons;  
  std::vector<L1CaloEmCand> m_nonIsoElectrons;
  unsigned m_mipBits;   //IMPORTANT - the static_casts in the getCables1And2 method should
  unsigned m_quietBits; //match the types of m_mipBits and m_quietBits.
  std::vector<L1CaloRegion> m_regions;
  
    
};

std::ostream& operator << (std::ostream& os, const L1GctSourceCard& card);

#endif /*L1GCTSOURCECARD_H_*/
