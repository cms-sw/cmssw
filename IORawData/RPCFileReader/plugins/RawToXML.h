#ifndef IORawData_RPCFileReader_RawToXML_H
#define IORawData_RPCFileReader_RawToXML_H

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace edm { class Event; class EventSetup; class ParameterSet; }
class XMLDataIO;


class RawToXML : public edm::EDAnalyzer {
public:
  explicit RawToXML(const edm::ParameterSet&);
  virtual ~RawToXML();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
private:
  edm::InputTag theDataLabel;
  XMLDataIO * theWriter; 
};
#endif

