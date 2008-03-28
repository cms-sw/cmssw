#ifndef L1GCTWHEELJETFPGA_H_
#define L1GCTWHEELJETFPGA_H_

/*!
* \class L1GctWheelJetFpga
* \brief Represents a GCT Wheel Jet FPGA
*
*  Takes as input the Jet and Ht data from one eta half of CMS
*  (three leaf cards of data) and summarises/reduces this data
*  before passing it onto the L1GctJetFinalStage processing that
*  takes place (physically) on the concentrator card.
* 
* \author Jim Brooke & Robert Frazier
* \date May 2006
*/ 

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctJetCand.h"
#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEtHad.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctJetLeafCard.h"

#include "L1Trigger/GlobalCaloTrigger/src/L1GctUnsignedInt.h"
#include "L1Trigger/GlobalCaloTrigger/src/L1GctJetCount.h"

class L1GctJetCounter;

#include <vector>

class L1GctWheelJetFpga : public L1GctProcessor
{
public:
  typedef std::vector<L1GctJetCand> JetVector;
  typedef L1GctUnsignedInt<L1GctEtHad::kEtHadNBits> EtHadType;
  typedef L1GctJetLeafCard::hfTowerSumsType hfTowerSumsType;

  /// Max number of jets of each type we output.
  static const int MAX_JETS_OUT;

  /// Max number of leaf card pointers
  static const unsigned int MAX_LEAF_CARDS;

  /// Max number of jets input from each leaf card
  static const unsigned int MAX_JETS_PER_LEAF;

  /// Number of jet counters
  static const unsigned int N_JET_COUNTERS;

  /// id must be 0 / 1 for -ve/+ve eta halves of CMS
  L1GctWheelJetFpga(int id,
		    std::vector<L1GctJetLeafCard*> inputLeafCards);

  /// destructor
  ~L1GctWheelJetFpga();

  /// Overload << operator
  friend std::ostream& operator << (std::ostream& os, const L1GctWheelJetFpga& fpga);

  /// clear internal buffers
  void reset();

  /// get input data from sources
  virtual void fetchInput();

  /// process the data, fill output buffers
  virtual void process();

  /// define the bunch crossing range to process
  void setBxRange(const int firstBx, const int numberOfBx);

  /// partially clear buffers
  void setNextBx(const int bx);

  /// set input data      
  void setInputJet(int i, L1GctJetCand jet); 
  void setInputHt (int i, unsigned ht);
    
  /// get the input jets. Jets 0-5 from leaf card 0, jetfinderA.  Jets 6-11 from leaf card 0, jetfinder B... etc.
  JetVector getInputJets() const { return m_inputJets; }
    
  /// get the input Ht
  EtHadType inputHt(unsigned leafnum) const { return m_inputHt.at(leafnum); }
    
  /// get the input Hf Sums
  hfTowerSumsType inputHfSums(unsigned leafnum) const { return m_inputHfSums.at(leafnum); }

  /// get the output jets
  JetVector getCentralJets() const { return m_centralJets; }

  /// get the output jets
  JetVector getForwardJets() const { return m_forwardJets; }

  /// get the output jets
  JetVector getTauJets() const { return m_tauJets; }
    
  /// get the output Ht
  EtHadType getOutputHt() const { return m_outputHt; }

  /// get the output Hf Sums
  hfTowerSumsType getOutputHfSums() const { return m_outputHfSums; }

  /// get the output jet counts
  L1GctJetCount<3> getOutputJc(unsigned jcnum) const
    { return ( (jcnum<N_JET_COUNTERS) ? m_outputJc.at(jcnum) : 0); }

  /// Get the jet counters
  L1GctJetCounter* getJetCounter(unsigned jcnum) const
    { return ( (jcnum<N_JET_COUNTERS) ? m_jetCounters.at(jcnum) : 0); }

 protected:

  /// Separate reset methods for the processor itself and any data stored in pipelines
  virtual void resetProcessor();
  virtual void resetPipelines() {}

  /// Initialise inputs with null objects for the correct bunch crossing if required
  virtual void setupObjects();

private:

  static const int MAX_JETS_IN;    ///< Maximum number of jets we can have as input
  static const int MAX_RAW_CJETS;  ///< Max. possible central jets to sort over
  static const int MAX_RAW_FJETS;  ///< Max. possible forward jets to sort over
  static const int MAX_RAW_TJETS;  ///< Max. possible tau jets to sort over
    
  /// algo ID
  int m_id;

  /// the jet leaf cards
  std::vector<L1GctJetLeafCard*> m_inputLeafCards;

  /// the jet counters
  std::vector<L1GctJetCounter*> m_jetCounters;
    
  /// input data. Jets 0-5 from leaf card 0, jetfinderA.  Jets 6-11 from leaf card 0, jetfinder B... etc.
  JetVector m_inputJets;
    
  // Holds the all the various inputted jets, re-addressed using proper GCT->GT jet addressing
  JetVector m_rawCentralJets; 
  JetVector m_rawForwardJets; 
  JetVector m_rawTauJets; 

  // input Ht sums from each leaf card
  std::vector< EtHadType > m_inputHt;

  // input Hf Et sums from each leaf card
  std::vector< hfTowerSumsType > m_inputHfSums;

  // output data
  JetVector m_centralJets;
  JetVector m_forwardJets;
  JetVector m_tauJets;
    
  // data sent to GlobalEnergyAlgos
  EtHadType m_outputHt;
  hfTowerSumsType m_outputHfSums;
  std::vector< L1GctJetCount<3> > m_outputJc;
      
  //PRIVATE METHODS
  /// Check the setup, independently of how we have been constructed
  void checkSetup();
  /// Puts the output from a jetfinder into the correct index range of the m_inputJets array. 
  void storeJets(JetVector jets, unsigned short iLeaf, unsigned short offset);
  /// Classifies jets into central, forward or tau.
  void classifyJets();
  /// Initialises all the jet vectors with jets of the correct type.
  void setupJetsVectors(const int16_t bx);
};

std::ostream& operator << (std::ostream& os, const L1GctWheelJetFpga& fpga);

#endif /*L1GCTWHEELJETFPGA_H_*/
