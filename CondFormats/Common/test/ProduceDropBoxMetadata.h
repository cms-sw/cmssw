#ifndef ProduceDropBoxMetadata_H
#define ProduceDropBoxMetadata_H

/** \class ProduceDropBoxMetadata
 *  No description available.
 *
 *  $Date: 2011/02/23 16:55:18 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - CERN
 */
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>
#include <string>

class ProduceDropBoxMetadata  : public edm::EDAnalyzer {
public:
  /// Constructor
  ProduceDropBoxMetadata(const edm::ParameterSet&);

  /// Destructor
  virtual ~ProduceDropBoxMetadata();

  // Operations
//   virtual void beginJob();
  virtual void beginRun(const edm::Run& run, const edm::EventSetup& eSetup);
  
  virtual void analyze(const edm::Event&, const edm::EventSetup&) {}
  virtual void endJob() {}

protected:

private:

  bool read;
  bool write;

  std::vector<edm::ParameterSet> fToWrite;
  std::vector<std::string> fToRead;

};
#endif

