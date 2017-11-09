//-------------------------------------------------
//
/**  \class DTTFFEDSim
 *
 *   L1 DT Track Finder Digi-to-Raw
 *
 *
 *
 *   J. Troconiz  UAM Madrid
 */
//
//--------------------------------------------------
#ifndef DTTFRawToDigi_DTTFFEDSim_h
#define DTTFRawToDigi_DTTFFEDSim_h

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include <DataFormats/L1DTTrackFinder/interface/L1MuDTChambPhContainer.h>
#include <DataFormats/L1DTTrackFinder/interface/L1MuDTChambThContainer.h>
#include <DataFormats/L1DTTrackFinder/interface/L1MuDTTrackContainer.h>

#include <FWCore/Framework/interface/stream/EDProducer.h>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <FWCore/Utilities/interface/InputTag.h>

#include <string>

class DTTFFEDSim : public edm::stream::EDProducer<> {

 public:

  /// Constructor
  DTTFFEDSim(const edm::ParameterSet& pset);

  /// Destructor
  ~DTTFFEDSim() override;

  /// Produce digis out of raw data
  void produce(edm::Event & e, const edm::EventSetup& c) override;

  /// Generate and fill FED raw data for a full event
  bool fillRawData(edm::Event& e,
                   FEDRawDataCollection& data);

 private:
  
  unsigned int eventNum;

  edm::InputTag DTDigiInputTag;
  edm::InputTag DTPHTFInputTag;

 // utilities
  int channel(int wheel, int sector, int bx);

  int bxNr(int channel);

  int sector(int channel);

  int wheel(int channel);

  void calcCRC(int myD1, int myD2, int &myC);

  edm::InputTag getDTDigiInputTag() { return DTDigiInputTag; }
  edm::InputTag getDTPHTFInputTag() { return DTPHTFInputTag; }

  edm::EDGetTokenT<L1MuDTChambPhContainer> ChPh_tok;
  edm::EDGetTokenT<L1MuDTChambThContainer> ChTh_tok;
  edm::EDGetTokenT<L1MuDTTrackContainer>   Trk_tok;


};
#endif
