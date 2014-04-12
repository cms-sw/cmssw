#include "EvFFEDSelector.h"

namespace evf{

  EvFFEDSelector::EvFFEDSelector( const edm::ParameterSet& ps)
    : label_(ps.getParameter<edm::InputTag>("inputTag"))
    , fedlist_(ps.getParameter<std::vector<unsigned int> >("fedList")) 
  {
    token_ = consumes<FEDRawDataCollection>(label_);
    produces<FEDRawDataCollection>();
  }
  void EvFFEDSelector::produce(edm::Event & e, const edm::EventSetup& c)
  {
    edm::Handle<FEDRawDataCollection> rawdata;
    FEDRawDataCollection *fedcoll = new FEDRawDataCollection();
    e.getByToken(token_,rawdata);
    std::vector<unsigned int>::iterator it = fedlist_.begin();
    for(;it!=fedlist_.end();it++)
      {
	const FEDRawData& data = rawdata->FEDData(*it);
	if(data.size()>0){
	  FEDRawData& fedData=fedcoll->FEDData(*it);
	  fedData.resize(data.size());
	  memcpy(fedData.data(),data.data(),data.size());
	} 
      }
    std::auto_ptr<FEDRawDataCollection> bare_product(fedcoll);
    e.put(bare_product);
  }
}
