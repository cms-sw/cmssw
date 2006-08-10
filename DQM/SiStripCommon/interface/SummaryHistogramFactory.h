#ifndef DQM_SiStripCommon_SummaryHistogramFactory_H
#define DQM_SiStripCommon_SummaryHistogramFactory_H

#include "DQM/SiStripCommon/interface/SiStripEnumeratedTypes.h"
#include "DQM/SiStripCommon/interface/SummaryGenerator.h"
#include <boost/cstdint.hpp>
#include "TH1.h"
#include <string>
#include <map>

template<class T>
class SummaryHistogramFactory {
  
 public:
  
  void generate( const sistrip::SummaryHisto& histo, 
		 const sistrip::SummaryType& type,
		 const sistrip::View& view, 
		 const std::string& directory, 
		 const std::map<uint32_t,T>& data,
		 TH1& summary_histo );
  
 private:
  
  void fill( const sistrip::SummaryHisto& histo, 
	     const sistrip::View& view, 
	     const uint32_t& key,
	     const std::map<uint32_t,T>& data,
	     std::auto_ptr<SummaryGenerator> generator );
  
  void format( const sistrip::SummaryHisto& histo, 
	       const sistrip::SummaryType& type,
	       const sistrip::View& view, 
	       const uint32_t& key,
	       TH1& summary_histo );
  
};

#endif // DQM_SiStripCommon_SummaryHistogramFactory_H


