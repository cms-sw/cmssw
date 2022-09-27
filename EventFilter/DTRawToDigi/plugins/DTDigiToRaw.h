#ifndef EventFilter_DTDigiToRaw_h
#define EventFilter_DTDigiToRaw_h

#include <DataFormats/DTDigi/interface/DTDigiCollection.h>

class FEDRawDataCollection;

#include <CondFormats/DTObjects/interface/DTReadOutMapping.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <DataFormats/FEDRawData/interface/FEDRawData.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <map>

class DTDigiToRaw {
public:
  /// Constructor
  DTDigiToRaw(const edm::ParameterSet& ps);

  /// Destructor
  virtual ~DTDigiToRaw();

  /// Take a vector of digis and fill the FEDRawDataCollection
  FEDRawData* createFedBuffers(const DTDigiCollection& digis, edm::ESHandle<DTReadOutMapping>& mapping);

  void SetdduID(int dduid);

private:
  typedef unsigned int Word32;
  typedef long long Word64;
  const edm::ParameterSet pset;

  int dduID_;
  bool debug;
};
#endif
