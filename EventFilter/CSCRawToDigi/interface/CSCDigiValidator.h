/**\class CSCDigiValidator CSCDigiValidator.cc UserCode/CSCDigiValidator/src/CSCDigiValidator.cc
*/

// Original Author:  Lindsey Gray
//         Created:  Tue Jul 28 18:04:11 CEST 2009

#include <memory>
#include <string>

#include <FWCore/Framework/interface/ConsumesCollector.h>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCStripDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigiCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/CSCTriggerContainer.h"
#include "DataFormats/L1CSCTrackFinder/interface/L1CSCTrackCollection.h"
#include "DataFormats/L1CSCTrackFinder/interface/TrackStub.h"

class CSCWireDigi;
class CSCStripDigi;
class CSCComparatorDigi;
class CSCCLCTDigi;
class CSCALCTDigi;
class CSCCorrelatedLCTDigi;
class CSCDetId;
class CSCChamberMap;

class CSCDigiValidator : public edm::EDFilter {
   public:
      explicit CSCDigiValidator(const edm::ParameterSet&);
      ~CSCDigiValidator() override;

   private:
      void beginJob() override ;
      bool filter(edm::Event&, const edm::EventSetup&) override;
      void endJob() override ;
      
      std::vector<CSCWireDigi> 
	sanitizeWireDigis(std::vector<CSCWireDigi>::const_iterator,
			  std::vector<CSCWireDigi>::const_iterator);
      std::vector<CSCStripDigi>
	relabelStripDigis(const CSCChamberMap*,CSCDetId,
			  std::vector<CSCStripDigi>::const_iterator,
			  std::vector<CSCStripDigi>::const_iterator);
      std::vector<CSCStripDigi>
	sanitizeStripDigis(std::vector<CSCStripDigi>::const_iterator,
			   std::vector<CSCStripDigi>::const_iterator);
      std::vector<CSCStripDigi>
	zeroSupStripDigis(std::vector<CSCStripDigi>::const_iterator,
			  std::vector<CSCStripDigi>::const_iterator);
      std::vector<CSCComparatorDigi> 
	relabelCompDigis(const CSCChamberMap* m, CSCDetId _id,
			 std::vector<CSCComparatorDigi>::const_iterator b,
			 std::vector<CSCComparatorDigi>::const_iterator e);
      std::vector<CSCComparatorDigi>
	zeroSupCompDigis(std::vector<CSCComparatorDigi>::const_iterator,
			 std::vector<CSCComparatorDigi>::const_iterator);

      // ----------member data ---------------------------
      edm::InputTag wire1,strip1,comp1,clct1,alct1,lct1,csctf1,csctfstubs1;
      edm::InputTag wire2,strip2,comp2,clct2,alct2,lct2,csctf2,csctfstubs2;

      bool reorderStrips;

      edm::EDGetTokenT<CSCWireDigiCollection>                   wd1_token;
      edm::EDGetTokenT<CSCStripDigiCollection>                  sd1_token;
      edm::EDGetTokenT<CSCComparatorDigiCollection>             cd1_token;
      edm::EDGetTokenT<CSCALCTDigiCollection>                   al1_token;
      edm::EDGetTokenT<CSCCLCTDigiCollection>                   cl1_token;
      edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection>          co1_token;
      edm::EDGetTokenT<L1CSCTrackCollection>                    tr1_token;
      edm::EDGetTokenT<CSCTriggerContainer<csctf::TrackStub> >  ts1_token;

      edm::EDGetTokenT<CSCWireDigiCollection>                   wd2_token;
      edm::EDGetTokenT<CSCStripDigiCollection>                  sd2_token;
      edm::EDGetTokenT<CSCComparatorDigiCollection>             cd2_token;
      edm::EDGetTokenT<CSCALCTDigiCollection>                   al2_token;
      edm::EDGetTokenT<CSCCLCTDigiCollection>                   cl2_token;
      edm::EDGetTokenT<CSCCorrelatedLCTDigiCollection>          co2_token;
      edm::EDGetTokenT<L1CSCTrackCollection>                    tr2_token;
      edm::EDGetTokenT<CSCTriggerContainer<csctf::TrackStub> >  ts2_token;
};
