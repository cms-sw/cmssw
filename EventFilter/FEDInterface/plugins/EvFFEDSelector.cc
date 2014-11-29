#include <memory>

#include "EvFFEDSelector.h"

namespace evf {

  EvFFEDSelector::EvFFEDSelector(edm::ParameterSet const & config) :
    token_( consumes<FEDRawDataCollection>( config.getParameter<edm::InputTag>("inputTag")) ),
    fedlist_( config.getParameter<std::vector<unsigned int> >("fedList") )
  {
    produces<FEDRawDataCollection>();
  }


  void EvFFEDSelector::produce(edm::StreamID sid, edm::Event & event, edm::EventSetup const & setup) const
  {
    edm::Handle<FEDRawDataCollection> rawdata;
    event.getByToken(token_, rawdata);

    std::unique_ptr<FEDRawDataCollection> fedcoll( new FEDRawDataCollection() );

    for (unsigned int i : fedlist_)
      if (rawdata->FEDData(i).size() > 0)
        fedcoll->FEDData(i) = rawdata->FEDData(i);

    event.put(std::move(fedcoll));
  }

} // namespace evf
