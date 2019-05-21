//-------------------------------------------------
//
/**  \class DTTM7FEDReader
 *
 *   L1 DT TwinMux Raw-to-Digi
 *
 *
 *
 *   C. F. Bedoya -- CIEMAT
 *   G. Codispoti -- INFN Bologna
 *   J. Pazzini   -- INFN Padova
 */
//
//--------------------------------------------------
#ifndef L1TXRAWTODIGI_L1TTWINMUXRAWTODIGI_HH
#define L1TXRAWTODIGI_L1TTWINMUXRAWTODIGI_HH

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h"
#include "DataFormats/L1DTTrackFinder/interface/L1MuDTTrackContainer.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <string>

class L1TTwinMuxRawToDigi : public edm::global::EDProducer<> {
public:
  /// Constructor
  L1TTwinMuxRawToDigi(const edm::ParameterSet& pset);

  /// Destructor
  ~L1TTwinMuxRawToDigi() override;

  /// Produce digis out of raw data
  void produce(edm::StreamID, edm::Event& e, const edm::EventSetup& c) const override;

  /// Generate and fill FED raw data for a full event
  bool fillRawData(edm::Event& e,
                   L1MuDTChambPhContainer::Phi_Container& phi_data,
                   L1MuDTChambThContainer::The_Container& the_data,
                   L1MuDTChambPhContainer::Phi_Container& phi_out_data) const;

  void processFed(int twinmuxfed,
                  int wheel,
                  std::array<short, 12> const& twinMuxAmcSec,
                  edm::Handle<FEDRawDataCollection> data,
                  L1MuDTChambPhContainer::Phi_Container& phi_data,
                  L1MuDTChambThContainer::The_Container& the_data,
                  L1MuDTChambPhContainer::Phi_Container& phi_out_data) const;

private:
  bool debug_;
  size_t nfeds_;
  edm::InputTag DTTM7InputTag_;
  std::vector<int> feds_;
  std::vector<int> wheels_;
  std::vector<long long int> amcsecmap_;
  std::vector<std::array<short, 12> > amcsec_;

  // utilities
  inline unsigned char* readline(unsigned char* lineFED, int& lines, long& dataWord) const {
    dataWord = *((long*)lineFED);
    lineFED += 8;
    ++lines;
    return lineFED;
  }

  edm::InputTag getDTTM7InputTag() const { return DTTM7InputTag_; }

  edm::EDGetTokenT<FEDRawDataCollection> Raw_token;

  int normBx(int bx_, int bxCnt_) const;
  int radAngConversion(int radAng_) const;
  int benAngConversion(int benAng_) const;
};

#endif
