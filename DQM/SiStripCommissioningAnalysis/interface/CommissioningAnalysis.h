#ifndef DQM_SiStripCommissioningAnalysis_CommissioningAnalysis_H
#define DQM_SiStripCommissioningAnalysis_CommissioningAnalysis_H

#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripFedKey.h"
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
  
  CommissioningAnalysis( const uint32_t& key, 
			 const std::string& my_name );
  CommissioningAnalysis( const std::string& my_name );
  virtual ~CommissioningAnalysis() {;}
  
  typedef std::vector<float> VFloats;
  typedef std::vector<VFloats> VVFloats;
  typedef std::vector<uint16_t> VInts;
  typedef std::vector<VInts> VVInts;
  
  /** Extracts monitorables from TH1s for a given FED channel. */
  void analysis( const std::vector<TH1*>& );

  /** Identifies if analysis is valid or not. */
  virtual bool isValid() { return true; } //@@ should be pure virtual
  
  /** Prints monitorables */
  virtual void print( std::stringstream&, uint32_t ) = 0;
  
 protected:
  
  typedef std::pair<TH1*,std::string> Histo;
  inline const SiStripFecKey& fec() const;
  inline const SiStripFedKey& fed() const;
  inline const std::string& myName() const;
  void header( std::stringstream& ) const;
  void extractFedKey( const TH1* const );

 private:

  CommissioningAnalysis() {;}
  virtual void reset() = 0;
  virtual void extract( const std::vector<TH1*>& ) = 0;
  virtual void analyse() = 0;
  
  SiStripFecKey fec_;
  SiStripFedKey fed_;
  std::string myName_;

};

const SiStripFecKey& CommissioningAnalysis::fec() const { return fec_; }
const SiStripFedKey& CommissioningAnalysis::fed() const { return fed_; }
const std::string& CommissioningAnalysis::myName() const { return myName_; }

#endif // DQM_SiStripCommissioningAnalysis_CommissioningAnalysis_H

