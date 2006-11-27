#ifndef DQM_SiStripCommissioningAnalysis_CommissioningAnalysis_H
#define DQM_SiStripCommissioningAnalysis_CommissioningAnalysis_H

#include <boost/cstdint.hpp>
#include <sstream>
#include <string>
#include <vector>

class TH1;

/**
   @class CommissioningAnalysis
   @author M.Wingham, R.Bainbridge 

   @brief Abstract base for derived classes that provide analysis of
   commissioning histograms. Analysis is always performed at the level
   of a FED channel, so analysis() method takes vector of all
   TH1 histograms associated with a given FED channel. 
*/
class CommissioningAnalysis {

 public:
  
  CommissioningAnalysis( const uint32_t& key );
  CommissioningAnalysis();
  virtual ~CommissioningAnalysis() {;}
  
  typedef std::vector<float> VFloats;
  typedef std::vector<VFloats> VVFloats;
  typedef std::vector<uint16_t> VInts;
  typedef std::vector<VInts> VVInts;
  
  /** Extracts monitorables from TH1s for a given FED channel. */
  void analysis( const std::vector<TH1*>& );
  
  /** Prints monitorables */
  virtual void print( std::stringstream&, uint32_t ) = 0;
  
 protected:
  
  typedef std::pair<TH1*,std::string> Histo;
  inline uint32_t key() const;
  
 private:
  
  virtual void reset() = 0; 
  virtual void extract( const std::vector<TH1*>& ) = 0; 
  virtual void analyse() = 0; 
  
  uint32_t key_;
  
};

uint32_t CommissioningAnalysis::key() const { return key_; }

#endif // DQM_SiStripCommissioningAnalysis_CommissioningAnalysis_H

