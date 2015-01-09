#include <memory>

#include "EvFFEDSelector.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>

namespace evf {

  EvFFEDSelector::EvFFEDSelector(edm::ParameterSet const & config) :
    token_( consumes<FEDRawDataCollection>( config.getParameter<edm::InputTag>("inputTag")) ),
    fedlist_( config.getParameter<std::vector<unsigned int> >("fedList") )
  {
    produces<FEDRawDataCollection>();
  }

  void EvFFEDSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("inputTag",edm::InputTag("source"));
    {
      std::vector<unsigned int> temp1;
      temp1.reserve(2);
      temp1.push_back(812);
      temp1.push_back(1023);
      desc.add<std::vector<unsigned int> >("fedList",temp1);
    }
    descriptions.add("EvFFEDSelector",desc);
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
