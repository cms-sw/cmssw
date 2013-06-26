/**\class CSCDigiValidator CSCDigiValidator.cc UserCode/CSCDigiValidator/src/CSCDigiValidator.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Lindsey Gray
//         Created:  Tue Jul 28 18:04:11 CEST 2009
// $Id: CSCDigiValidator.h,v 1.1 2009/11/09 20:29:43 lgray Exp $
//
//


// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/InputTag.h"

//
// class decleration
//

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
      ~CSCDigiValidator();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
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
};

