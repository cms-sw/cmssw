#ifndef L1GCTELECTRONSORTER_H_
#define L1GCTELECTRONSORTER_H_

#include "DataFormats/L1CaloTrigger/interface/L1CaloEmCand.h"

#include "DataFormats/L1GlobalCaloTrigger/interface/L1GctEmCand.h"
#include "L1Trigger/GlobalCaloTrigger/interface/L1GctProcessor.h"

#include <vector>


/*!
 * \Class L1GctElectronSorter
 * \brief Class that sorts electron candidates
 *
 * This class can be constructed to sort iso or non-iso
 * electron candidates.
 * The electrons are sorted in ascending order and the 4
 * highest in rank will be returned.
 * It represents the 1st stage sorter FPGA's on the electron
 * leaf cards.
 *
 * \author  Maria Hansen
 * \date    21/04/06
 */

class L1GctElectronSorter : public L1GctProcessor
{
 public:

  /// Data type to associate each electron candidate with
  /// a priority based on its position in the sorting tree.
  /// Priority is used (as in the hardware) to decide which 
  /// electron is preferred when they have equal rank.

  struct prioritisedEmCand {
    L1GctEmCand emCand;
    unsigned short priority;

    // Define some constructors
    prioritisedEmCand() : emCand(), priority(0) {}
    prioritisedEmCand(L1GctEmCand& c, unsigned short p)  : emCand(c), priority(p) {}
    prioritisedEmCand(L1CaloEmCand& c, unsigned short p) : emCand(c), priority(p) {}

    // Enable some methods
    unsigned rank() const { return emCand.rank(); }

  };
  
  /// comparison operator for sort, used here and in the ElectronFinalSort
  /// Candidates of equal rank are sorted by priority, with the lower value given precedence
  static bool rankByGt(const prioritisedEmCand& x, const prioritisedEmCand& y) {
    return ( x.rank() > y.rank() || ( (x.rank() == y.rank()) && (x.priority < y.priority) ) ) ;
  }

  /// constructor; set type (isolated or non-isolated)
  L1GctElectronSorter(int nInputs, bool iso);
  ///   
  ~L1GctElectronSorter() override;
  ///
  /// get input data from sources
  void fetchInput() override;
  ///
  /// process the data, fill output buffers
  void process() override;
  ///
  /// set input candidate
  void setInputEmCand(const L1CaloEmCand& cand);
  ///	
  /// get input candidates
  inline std::vector<L1CaloEmCand> getInputCands() { return m_inputCands; }
  ///
  /// get output candidates
  inline std::vector<L1GctEmCand> getOutputCands() { return m_outputCands; }
  ///
  /// overload of cout operator
  friend std::ostream& operator<<(std::ostream& s,const L1GctElectronSorter& card);  
 protected:

  /// Separate reset methods for the processor itself and any data stored in pipelines
  void resetProcessor() override;
  void resetPipelines() override {}

  /// Initialise inputs with null objects for the correct bunch crossing if required
  void setupObjects() override;

 private:
  ///
  /// algo ID (is it FPGA 1 or 2 processing)
  int m_id;
  ///
  /// type of electron to sort (isolated = 0 or non isolated = 1)
  bool m_isolation;
  ///
  /// input data
  std::vector<L1CaloEmCand> m_inputCands;
  ///
  /// output data
  std::vector<L1GctEmCand> m_outputCands;
  
};

std::ostream& operator<<(std::ostream& s,const L1GctElectronSorter& card); 
#endif /*L1GCTELECTRONSORTER_H_*/
