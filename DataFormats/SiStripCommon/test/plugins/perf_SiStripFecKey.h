// Last commit: $Id: perf_SiStripFecKey.h,v 1.2 2010/01/07 11:20:48 lowette Exp $

#ifndef DataFormats_SiStripCommon_perfSiStripFecKey_H
#define DataFormats_SiStripCommon_perfSiStripFecKey_H

#include "DataFormats/SiStripCommon/interface/SiStripFecKey.h"
#include "DataFormats/SiStripCommon/interface/SiStripKey.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include <boost/cstdint.hpp>
#include <vector>

/**
   @class perfSiStripFecKey 
   @author R.Bainbridge
   @brief Simple class that tests SiStripFecKey.
*/
class perfSiStripFecKey : public edm::EDAnalyzer {
  
 public:
  
  perfSiStripFecKey( const edm::ParameterSet& );
  ~perfSiStripFecKey();
  
  void beginJob();
  void analyze( const edm::Event&, const edm::EventSetup& );
  void endJob() {;}

 private:
  
  class Value {
  public:
    uint16_t crate_, slot_, ring_, ccu_, module_, lld_, i2c_;
    Value() : crate_(0), slot_(0), ring_(0), ccu_(0), module_(0), lld_(0), i2c_(0) {;}
    Value( uint16_t crate, uint16_t slot, uint16_t ring, uint16_t ccu, uint16_t module, uint16_t lld, uint16_t i2c ) : 
      crate_(crate), slot_(slot), ring_(ring), ccu_(ccu), module_(module), lld_(lld), i2c_(i2c) {;}
  };
  
  void build( std::vector<Value>&,
	      std::vector<uint32_t>&,
	      std::vector<std::string>&,
	      std::vector<SiStripFecKey>&,
	      std::vector<SiStripKey>& );
  
  void build( const std::vector<Value>& ) const;
  void build( const std::vector<uint32_t>& ) const;
  void build( const std::vector<std::string>& ) const;
  void build( const std::vector<SiStripFecKey>& ) const;
  void build( const std::vector<SiStripKey>& ) const;

  void test( const std::vector<SiStripFecKey>& ) const;

  uint32_t loops_;
  
};

#endif // DataFormats_SiStripCommon_perfSiStripFecKey_H

