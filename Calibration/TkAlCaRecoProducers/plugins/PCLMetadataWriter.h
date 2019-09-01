#ifndef PCLMetadataWriter_H
#define PCLMetadataWriter_H

/** \class PCLMetadataWriter
 *  No description available.
 *
 *  \author G. Cerminara - CERN
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <vector>

class PCLMetadataWriter : public edm::EDAnalyzer {
public:
  /// Constructor
  PCLMetadataWriter(const edm::ParameterSet &);

  /// Destructor
  ~PCLMetadataWriter() override;

  // Operations
  //   virtual void beginJob            (void);
  //   virtual void endJob              (void);
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void beginRun(const edm::Run &, const edm::EventSetup &) override;
  void endRun(const edm::Run &, const edm::EventSetup &) override;
  //   virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const
  //   edm::EventSetup&); virtual void endLuminosityBlock  (const
  //   edm::LuminosityBlock&, const edm::EventSetup&);

protected:
private:
  bool readFromDB;
  std::map<std::string, std::map<std::string, std::string>> recordMap;
};
#endif
