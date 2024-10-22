#ifndef GTPSBTEXTTODIGI_H
#define GTPSBTEXTTODIGI_H

/*\class GtPsbTextToDigi
 *\description makes digis from GT PSB captured data file
 *\author Nuno Leonardo (CERN)
 *\date 08.08
 */

/*\note on format
  input file names: m_textFileName + mem# + .txt
  each line corresponds to one 80MHz clock (2 cycles/lines per event)
  input data: 16 bits: cycle (msb) + GCT em cand raw data (14:0)
  msb set once at cycle 0 to indicate BC0 signal, for GT synch check
  file/mem#:0  line/cycle:0  electron 1
               line/cycle:1  electron 3
  file/mem#:1  line/cycle:0  electron 2
               line/cycle:1  electron 4
  file/mem#:6  line/cycle:0  electron 1
               line/cycle:1  electron 3
  file/mem#:7  line/cycle:0  electron 2
               line/cycle:1  electron 4
  as specified to me by I.Mikulec, M.Jeitler, J.Brooke
*/

#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <fstream>
#include <memory>

class GtPsbTextToDigi : public edm::one::EDProducer<> {
public:
  explicit GtPsbTextToDigi(const edm::ParameterSet &);
  ~GtPsbTextToDigi() override;

private:
  void produce(edm::Event &, const edm::EventSetup &) override;
  void endJob() override;

private:
  /// Create empty digi collection
  void putEmptyDigi(edm::Event &);

  /// Number of events to be offset wrt input
  int m_fileEventOffset;

  /// Name out input file
  std::string m_textFileName;

  /// Event counter
  int m_nevt;

  /// File handle
  std::ifstream m_file[4];

  /// Hold detected BC0 signal position per file
  int m_bc0[4];
};

#endif
