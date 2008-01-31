#ifndef L1GCTJETCOUNTER_H_
#define L1GCTJETCOUNTER_H_

#include "CondFormats/L1TObjects/interface/L1GctJetCounterSetup.h"

#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"
#include "L1Trigger/GlobalCaloTrigger/src/L1GctJetCount.h"

#include <boost/cstdint.hpp> //for uint16_t
#include <vector>

/*!
 * \class L1GctJetCounter
 * \brief Counts jets in one Wheel that pass criteria encoded in a JetCounterLut
 *
 * The actual contents of this class are fairly simple, since
 * all the real work is done elsewhere.
 *  
 * \author Greg Heath
 * \date June 2006
 */

class L1GctJetCounterLut;
class L1GctJetCand;
class L1GctJetLeafCard;

class L1GctJetCounter : public L1GctProcessor
{
public:
  //Typedefs
  typedef std::vector<L1GctJetCand> JetVector;

  //Statics
  static const unsigned int MAX_JETLEAF_CARDS;
  static const unsigned int MAX_JETS_PER_LEAF;
  static const unsigned int MAX_JETS_TO_COUNT;
    
  /// id needs to encode Wheel and jet count numbers
  L1GctJetCounter(int id, std::vector<L1GctJetLeafCard*> leafCards,
                  L1GctJetCounterLut* jetCounterLut=0);
                 
  ~L1GctJetCounter();
   
  /// Overload << operator
  friend std::ostream& operator << (std::ostream& os, const L1GctJetCounter& algo);

  /// clear internal buffers
  virtual void reset();

  /// get input data from sources
  virtual void fetchInput();

  /// process the data, fill output buffers
  virtual void process();

  /// set a new lut for this counter
  void setLut(const L1GctJetCounterLut& lut);

  /// set a new lut for this counter by specifying the cuts - just one cut
  void setLut(const L1GctJetCounterSetup::cutDescription& cut);

  /// set a new lut for this counter by specifying the cuts - list of cuts
  void setLut(const L1GctJetCounterSetup::cutsListForJetCounter& cutList);

  /// set the input jets (for test purposes)
  void setJets(JetVector& jets);

  /// get the JetCounterLut
  L1GctJetCounterLut* getJetCounterLut() const { return m_jetCounterLut; }

  /// get the jets
  JetVector getJets() const { return m_jets; }

  /// get the value of the counter, for input into the jet count sums
  L1GctJetCount<3> getValue() const { return m_value;}

private:

  //Statics

  /// algo ID
  int m_id;

  /// Jet Leaf Card pointers
  std::vector<L1GctJetLeafCard*> m_jetLeafCards;
	
  /// Jet Et Converstion LUT pointer
  L1GctJetCounterLut* m_jetCounterLut;

  /// The jets to be counted
  JetVector m_jets;
    
  /// The value of the counter
  L1GctJetCount<3> m_value;

  //PRIVATE METHODS
  
};

std::ostream& operator << (std::ostream& os, const L1GctJetCounter& algo);

#endif /*L1GCTJETCOUNTER_H_*/
