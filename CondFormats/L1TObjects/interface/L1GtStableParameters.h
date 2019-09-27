#ifndef CondFormats_L1TObjects_L1GtStableParameters_h
#define CondFormats_L1TObjects_L1GtStableParameters_h

/**
 * \class L1GtStableParameters
 * 
 * 
 * Description: L1 GT stable parameters.  
 *
 * Implementation:
 *    <TODO: enter implementation details>
 *   
 * \author: Vasile Mihai Ghete - HEPHY Vienna
 * 
 * $Date$
 * $Revision$
 *
 */

// system include files
#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>

#include <ostream>

// user include files
//   base class

// forward declarations

// class declaration
class L1GtStableParameters {
public:
  // constructor
  L1GtStableParameters();

  // destructor
  virtual ~L1GtStableParameters();

public:
  /// get / set the number of physics trigger algorithms
  inline unsigned int gtNumberPhysTriggers() const { return m_numberPhysTriggers; }

  void setGtNumberPhysTriggers(const unsigned int&);

  /// get / set the additional number of physics trigger algorithms
  inline unsigned int gtNumberPhysTriggersExtended() const { return m_numberPhysTriggersExtended; }

  void setGtNumberPhysTriggersExtended(const unsigned int&);

  /// get / set the number of technical triggers
  inline unsigned int gtNumberTechnicalTriggers() const { return m_numberTechnicalTriggers; }

  void setGtNumberTechnicalTriggers(const unsigned int&);

  ///  get / set the number of L1 muons received by GT
  inline unsigned int gtNumberL1Mu() const { return m_numberL1Mu; }

  void setGtNumberL1Mu(const unsigned int&);

  ///  get / set the number of L1 e/gamma objects received by GT
  inline unsigned int gtNumberL1NoIsoEG() const { return m_numberL1NoIsoEG; }

  void setGtNumberL1NoIsoEG(const unsigned int&);

  ///  get / set the number of L1 isolated e/gamma objects received by GT
  inline unsigned int gtNumberL1IsoEG() const { return m_numberL1IsoEG; }

  void setGtNumberL1IsoEG(const unsigned int&);

  ///  get / set the number of L1 central jets received by GT
  inline unsigned int gtNumberL1CenJet() const { return m_numberL1CenJet; }

  void setGtNumberL1CenJet(const unsigned int&);

  ///  get / set the number of L1 forward jets received by GT
  inline unsigned int gtNumberL1ForJet() const { return m_numberL1ForJet; }

  void setGtNumberL1ForJet(const unsigned int&);

  ///  get / set the number of L1 tau jets received by GT
  inline unsigned int gtNumberL1TauJet() const { return m_numberL1TauJet; }

  void setGtNumberL1TauJet(const unsigned int&);

  ///  get / set the number of L1 jet counts received by GT
  inline unsigned int gtNumberL1JetCounts() const { return m_numberL1JetCounts; }

  void setGtNumberL1JetCounts(const unsigned int&);

  /// hardware stuff

  ///   get / set the number of condition chips in GTL
  inline unsigned int gtNumberConditionChips() const { return m_numberConditionChips; }

  void setGtNumberConditionChips(const unsigned int&);

  ///   get / set the number of pins on the GTL condition chips
  inline unsigned int gtPinsOnConditionChip() const { return m_pinsOnConditionChip; }

  void setGtPinsOnConditionChip(const unsigned int&);

  ///   get / set the correspondence "condition chip - GTL algorithm word"
  ///   in the hardware
  inline const std::vector<int>& gtOrderConditionChip() const { return m_orderConditionChip; }

  void setGtOrderConditionChip(const std::vector<int>&);

  ///   get / set the number of PSB boards in GT
  inline int gtNumberPsbBoards() const { return m_numberPsbBoards; }

  void setGtNumberPsbBoards(const int&);

  ///   get / set the number of bits for eta of calorimeter objects
  inline unsigned int gtIfCaloEtaNumberBits() const { return m_ifCaloEtaNumberBits; }

  void setGtIfCaloEtaNumberBits(const unsigned int&);

  ///   get / set the number of bits for eta of muon objects
  inline unsigned int gtIfMuEtaNumberBits() const { return m_ifMuEtaNumberBits; }

  void setGtIfMuEtaNumberBits(const unsigned int&);

  ///    get / set WordLength
  inline int gtWordLength() const { return m_wordLength; }

  void setGtWordLength(const int&);

  ///    get / set one UnitLength
  inline int gtUnitLength() const { return m_unitLength; }

  void setGtUnitLength(const int&);

  /// print all the L1 GT stable parameters
  void print(std::ostream&) const;

private:
  /// trigger decision

  /// number of physics trigger algorithms
  unsigned int m_numberPhysTriggers;

  /// additional number of physics trigger algorithms
  unsigned int m_numberPhysTriggersExtended;

  /// number of technical triggers
  unsigned int m_numberTechnicalTriggers;

  /// trigger objects

  /// muons
  unsigned int m_numberL1Mu;

  /// e/gamma and isolated e/gamma objects
  unsigned int m_numberL1NoIsoEG;
  unsigned int m_numberL1IsoEG;

  /// central, forward and tau jets
  unsigned int m_numberL1CenJet;
  unsigned int m_numberL1ForJet;
  unsigned int m_numberL1TauJet;

  /// jet counts
  unsigned int m_numberL1JetCounts;

private:
  /// hardware

  /// number of condition chips
  unsigned int m_numberConditionChips;

  /// number of pins on the GTL condition chips
  unsigned int m_pinsOnConditionChip;

  /// correspondence "condition chip - GTL algorithm word" in the hardware
  /// chip 2: 0 - 95;  chip 1: 96 - 128 (191)
  std::vector<int> m_orderConditionChip;

  /// number of PSB boards in GT
  int m_numberPsbBoards;

  /// number of bits for eta of calorimeter objects
  unsigned int m_ifCaloEtaNumberBits;

  /// number of bits for eta of muon objects
  unsigned int m_ifMuEtaNumberBits;

private:
  /// GT DAQ record organized in words of WordLength bits
  int m_wordLength;

  /// one unit in the word is UnitLength bits
  int m_unitLength;

  COND_SERIALIZABLE;
};

#endif /*CondFormats_L1TObjects_L1GtStableParameters_h*/
