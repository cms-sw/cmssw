#ifndef OptAlignDataConverter_h
#define OptAlignDataConverter_h
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "CondTools/Utilities/interface/CSVFieldMap.h"
#include <string>
#include <vector>
namespace edm{
  class ParameterSet;
  class Event;
  class EventSetup;
}
class OptAlignDataConverter : public edm::EDAnalyzer {
 public:
  explicit OptAlignDataConverter(const edm::ParameterSet& iConfig );
  ~OptAlignDataConverter(){}
  virtual void analyze( const edm::Event&, const edm::EventSetup& ){}
  virtual void endJob();
 private:
  CSVFieldMap m_fieldMap;
  std::string m_inFileName;
};
#endif
