#ifndef DQM_SiStripCommissioningAnalysis_CommissioningAlgorithm_H
#define DQM_SiStripCommissioningAnalysis_CommissioningAlgorithm_H

#include <vector>
#include <string>
#include <cstdint>

class CommissioningAlgorithm;
class CommissioningAnalysis;
class TH1;

/**
   @class CommissioningAlgorithm
   @author R.Bainbridge 
   @brief 
*/
class CommissioningAlgorithm {
public:
  CommissioningAlgorithm(CommissioningAnalysis* const);

  CommissioningAlgorithm();

  virtual ~CommissioningAlgorithm() { ; }

  typedef std::pair<TH1*, std::string> Histo;

  /** Performs histogram analysis. */
  void analysis(const std::vector<TH1*>&);

protected:
  /** Extracts FED key from histogram title. */
  uint32_t extractFedKey(const TH1* const);

  /** Extracts and organises histograms. */
  virtual void extract(const std::vector<TH1*>&) = 0;

  /** Performs histogram anaylsis. */
  virtual void analyse() = 0;

  /** Analysis class. */
  inline CommissioningAnalysis* const anal() const;

private:
  /** Analysis class. */
  CommissioningAnalysis* anal_;
};

// ---------- inline methods ----------

CommissioningAnalysis* const CommissioningAlgorithm::anal() const { return anal_; }

#endif  // DQM_SiStripCommissioningAnalysis_CommissioningAlgorithm_H
