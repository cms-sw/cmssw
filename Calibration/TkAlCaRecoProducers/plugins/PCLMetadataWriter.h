#ifndef PCLMetadataWriter_H
#define PCLMetadataWriter_H

/** \class PCLMetadataWriter
 *  No description available.
 *
 *  $Date: 2011/03/03 09:32:46 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - CERN
 */

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <vector>
#include <string>


class PCLMetadataWriter  : public edm::EDAnalyzer {
public:
  /// Constructor
  PCLMetadataWriter(const edm::ParameterSet&);

  /// Destructor
  virtual ~PCLMetadataWriter();

  // Operations
//   virtual void beginJob            (void);
//   virtual void endJob              (void);  
  virtual void analyze             (const edm::Event&          , const edm::EventSetup&);
  virtual void beginRun            (const edm::Run&            , const edm::EventSetup&);
  virtual void endRun              (const edm::Run&            , const edm::EventSetup&);
//   virtual void beginLuminosityBlock(const edm::LuminosityBlock&, const edm::EventSetup&);
//   virtual void endLuminosityBlock  (const edm::LuminosityBlock&, const edm::EventSetup&);

protected:

private:
  bool readFromDB;
  std::map<std::string,  std::map<std::string, std::string> > recordMap;
  

};
#endif

