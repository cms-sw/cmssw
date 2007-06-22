#include "EventFilter/ESDigiToRaw/interface/ESDigiToRawTB.h"

#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalDetId/interface/EcalDetIdCollections.h"

ESDigiToRawTB::ESDigiToRawTB(const edm::ParameterSet& ps)
{

  label_ = ps.getParameter<string>("Label");
  instanceName_ = ps.getParameter<string>("InstanceES");
  debug_ = ps.getUntrackedParameter<bool>("debugMode", false);

  counter_ = 0;

  produces<FEDRawDataCollection>();

  ESDataFormatter_ = new ESDataFormatter(ps);

}

ESDigiToRawTB::~ESDigiToRawTB() {
  delete ESDataFormatter_;
}

void ESDigiToRawTB::beginJob(const edm::EventSetup& es) {
}

void ESDigiToRawTB::produce(edm::Event& ev, const edm::EventSetup& es) {

  run_number_ = ev.id().run();
  orbit_number_ = counter_ / BXMAX;
  bx_ = (counter_ % BXMAX);
  lv1_ = counter_;
  counter_++;

  ESDataFormatter_->setRunNumber(run_number_);
  ESDataFormatter_->setOrbitNumber(orbit_number_);
  ESDataFormatter_->setBX(bx_);
  ESDataFormatter_->setLV1(lv1_);

  pair<int,int> ESFEDIds = FEDNumbering::getPreShowerFEDIds();

  edm::Handle<ESDigiCollection> digis;
  ev.getByLabel(label_, instanceName_, digis);

  ESDataFormatter::Digis Digis;
  Digis.clear();

  int dccId = 0;
  for (ESDigiCollection::const_iterator it=digis->begin(); it!=digis->end(); ++it) {

    const ESDataFrame& df = *it;
    const ESDetId& detId = it->id();

    // Only select  19< iy < 22 and 30 < ix < 33 for TB for the time being
    if (detId.zside() == 1) {
      if (detId.six() >= 30 && detId.six() <= 33) {
	if (detId.siy() >= 19 && detId.siy() <= 22) {
	  int fedId = ESFEDIds.first + dccId;	    
	  Digis[fedId].push_back(df);	    
	  
	}
      }
    }

  }
  
  auto_ptr<FEDRawDataCollection> productRawData( new FEDRawDataCollection );

  int nFED = 0;
  for (int fId=ESFEDIds.first; fId<=ESFEDIds.second; ++fId) {
    if (nFED == 0) {
      FEDRawData *rawData = ESDataFormatter_->DigiToRawTB(fId, Digis);
      FEDRawData& fedRawData = productRawData->FEDData(fId); 
      fedRawData = *rawData;
      if (debug_) cout<<"FED : "<<fId<<" Data size : "<<fedRawData.size()<<" (Bytes)"<<endl;
    }
    nFED++;
  } 

  ev.put(productRawData);

  return;
}

void ESDigiToRawTB::endJob() {
}
