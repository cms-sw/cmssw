

#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h"
#include "DataFormats/FEDRawData/interface/FEDRawData.h"
#include "DataFormats/FEDRawData/interface/FEDNumbering.h"

#include "DataFormats/EcalDigi/interface/ESDataFrame.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"


#include "EventFilter/ESDigiToRaw/interface/ESDigiToRaw.h"
#include "EventFilter/ESDigiToRaw/src/ESDataFormatterV1_1.h"
#include "EventFilter/ESDigiToRaw/src/ESDataFormatterV4.h"


using namespace std;
using namespace edm;

ESDigiToRaw::ESDigiToRaw(const edm::ParameterSet& ps) : ESDataFormatter_(0)
{
  
  label_ = ps.getParameter<string>("Label");
  instanceName_ = ps.getParameter<string>("InstanceES");
  debug_ = ps.getUntrackedParameter<bool>("debugMode", false);
  formatMajor_ = ps.getUntrackedParameter<int>("formatMajor",1);
  formatMinor_ = ps.getUntrackedParameter<int>("formatMinor",1);

  counter_ = 0;
  kchip_ec_ = 0; 
  kchip_bc_ = 0; 

  produces<FEDRawDataCollection>();

  if (formatMajor_==4 && formatMinor_==0) 
    ESDataFormatter_ = new ESDataFormatterV4(ps);
  else 
    ESDataFormatter_ = new ESDataFormatterV1_1(ps);

}

ESDigiToRaw::~ESDigiToRaw() {
  if (ESDataFormatter_) delete ESDataFormatter_;
}

void ESDigiToRaw::beginJob(const edm::EventSetup& es) {
}

void ESDigiToRaw::produce(edm::Event& ev, const edm::EventSetup& es) {

  run_number_ = ev.id().run();
  orbit_number_ = counter_ / LHC_BX_RANGE;
  bx_ = (counter_ % LHC_BX_RANGE);
   
  //lv1_ = counter_;
  lv1_ = ev.id().event();
  kchip_ec_ = (lv1_ % KCHIP_EC_RANGE); 
  kchip_bc_ = (counter_ % KCHIP_BC_RANGE);
  counter_++;

  ESDataFormatter_->setRunNumber(run_number_);
  ESDataFormatter_->setOrbitNumber(orbit_number_);
  ESDataFormatter_->setBX(bx_);
  ESDataFormatter_->setLV1(lv1_);
  ESDataFormatter_->setKchipBC(kchip_bc_);
  ESDataFormatter_->setKchipEC(kchip_ec_);

  pair<int,int> ESFEDIds = FEDNumbering::getPreShowerFEDIds();

  edm::Handle<ESDigiCollection> digis;
  ev.getByLabel(label_, instanceName_, digis);

  ESDataFormatter::Digis Digis;
  Digis.clear();

  for (ESDigiCollection::const_iterator it=digis->begin(); it!=digis->end(); ++it) {

    const ESDataFrame& df = *it;
    const ESDetId& detId = it->id();

    // Fake DCC-fed map, for the time being
    int dccId = 0;
    if (detId.zside() == 1) {
      if (detId.plane() == 1) {
	if (detId.six()<=20 && detId.siy()<=20) {
	  dccId = 0;
	} else if (detId.six()>=20 && detId.siy()<=20) {
	  dccId = 1;
	} else if (detId.six()<=20 && detId.siy()>=20) {
	  dccId = 2;
	} else if (detId.six()>=20 && detId.siy()>=20) {
	  dccId = 3;
	}
      } else if (detId.plane() == 2) {
	if (detId.six()<=20 && detId.siy()<=20) {
	  dccId = 4;
	} else if (detId.six()>=20 && detId.siy()<=20) {
	  dccId = 5;
	} else if (detId.six()<=20 && detId.siy()>=20) {
	  dccId = 6;
	} else if (detId.six()>=20 && detId.siy()>=20) {
	  dccId = 7;
	}
      }
    } else if (detId.zside() == -1) {
      if (detId.plane() == 1) {
	if (detId.six()<=20 && detId.siy()<=20) {
	  dccId = 8;
	} else if (detId.six()>=20 && detId.siy()<=20) {
	  dccId = 9;
	} else if (detId.six()<=20 && detId.siy()>=20) {
	  dccId = 10;
	} else if (detId.six()>=20 && detId.siy()>=20) {
	  dccId = 11;
	}
      } else if (detId.plane() == 2) {
	if (detId.six()<=20 && detId.siy()<=20) {
	  dccId = 12;
	} else if (detId.six()>=20 && detId.siy()<=20) {
	  dccId = 13;
	} else if (detId.six()<=20 && detId.siy()>=20) {
	  dccId = 14;
	} else if (detId.six()>=20 && detId.siy()>=20) {
	  dccId = 15;
	}
      }
    }

    int fedId = ESFEDIds.first + dccId;

    Digis[fedId].push_back(df);
  }

  auto_ptr<FEDRawDataCollection> productRawData( new FEDRawDataCollection );

  for (int fId=ESFEDIds.first; fId<=ESFEDIds.second; ++fId) {
    FEDRawData *rawData = ESDataFormatter_->DigiToRaw(fId, Digis);
    FEDRawData& fedRawData = productRawData->FEDData(fId); 
    fedRawData = *rawData;
    if (debug_) cout<<"FED : "<<fId<<" Data size : "<<fedRawData.size()<<" (Bytes)"<<endl;
  } 

  ev.put(productRawData);

  return;
}

void ESDigiToRaw::endJob() {
}
