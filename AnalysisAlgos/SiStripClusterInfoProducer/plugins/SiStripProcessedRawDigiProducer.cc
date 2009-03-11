#include "AnalysisAlgos/SiStripClusterInfoProducer/plugins/SiStripProcessedRawDigiProducer.h"

SiStripProcessedRawDigiProducer::SiStripProcessedRawDigiProducer(edm::ParameterSet const& conf) : 
  conf_(conf),
  SiStripPedestalsSubtractor_(SiStripRawProcessingFactory::create_SubtractorPed(conf)),
  SiStripCommonModeNoiseSubtractor_(SiStripRawProcessingFactory::create_SubtractorCMN(conf))
{
  produces< edm::DetSetVector<SiStripProcessedRawDigi> >("");

  inputmap_["ProcessedRaw"]   = PR;  inputmap_["VirginRaw"]      = VR;
  inputmap_["ZeroSuppressed"] = ZS;  inputmap_["ScopeMode"]      = SM;
}

void 
SiStripProcessedRawDigiProducer::produce(edm::Event& e, const edm::EventSetup& es) {
  std::auto_ptr< edm::DetSetVector<SiStripProcessedRawDigi> > output(new edm::DetSetVector<SiStripProcessedRawDigi>());
  edm::Handle<   edm::DetSetVector<SiStripDigi>          > input_digi; 
  edm::Handle<   edm::DetSetVector<SiStripRawDigi>       > input_rawdigi; 

  es.get<SiStripGainRcd>().get(gainHandle_);
  
  Parameters DigiProducersList = conf_.getParameter<Parameters>("DigiProducersList");
  Parameters::const_iterator itDigiProducersList = DigiProducersList.begin();
  for(; itDigiProducersList != DigiProducersList.end(); ++itDigiProducersList ) {
    edm::InputTag input_tag = itDigiProducersList->getParameter<edm::InputTag>("DigiProducer");       

    //All 4 of the possible FEDOUTPUTs will be in an event, but only one will have size() nonzero.
    switch(inputmap_.find(input_tag.instance())->second) {
    case ZS:  e.getByLabel(input_tag, input_digi);    if(   input_digi->size()) { zs_process(*input_digi,    *output); }  break;
    case PR:  e.getByLabel(input_tag, input_rawdigi); if(input_rawdigi->size()) { pr_process(*input_rawdigi, *output); }  break;
    case SM:  //fall through to VR
    case VR:  e.getByLabel(input_tag, input_rawdigi); if(input_rawdigi->size()) {
      SiStripPedestalsSubtractor_->init(es);
      SiStripCommonModeNoiseSubtractor_->init(es);
      vr_process(*input_rawdigi, *output); 
    } break;
    default: throw(new cms::Exception("Unknown DigiProducer, error running SiStripProcessedRawDigiProducer"));
    }
  }
  e.put(output);
}

void 
SiStripProcessedRawDigiProducer::zs_process(const edm::DetSetVector<SiStripDigi> & input, edm::DetSetVector<SiStripProcessedRawDigi>& output) {
  for(edm::DetSetVector<SiStripDigi>::const_iterator DSV_it=input.begin(); DSV_it!=input.end(); DSV_it++)  {
    std::vector<float> digis;
    for(edm::DetSet<SiStripDigi>::const_iterator it=DSV_it->begin(); it!=DSV_it->end(); it++) {
      if(it->strip() + unsigned(1) > digis.size() ) { digis.resize(it->strip()+1, float(0.0)); }
      digis.at(it->strip())= static_cast<float>(it->adc());
    }
    common_process(digis, DSV_it->id, output);
  }
}

void 
SiStripProcessedRawDigiProducer::pr_process(const edm::DetSetVector<SiStripRawDigi> & input, edm::DetSetVector<SiStripProcessedRawDigi>& output) {
  for(edm::DetSetVector<SiStripRawDigi>::const_iterator DSV_it=input.begin(); DSV_it!=input.end(); DSV_it++) {
    std::vector<float> digis;
    transform(DSV_it->begin(), DSV_it->end(), back_inserter(digis), boost::bind(&SiStripRawDigi::adc , _1));
    common_process(digis, DSV_it->id, output);
  }
}

void 
SiStripProcessedRawDigiProducer::vr_process(const edm::DetSetVector<SiStripRawDigi> & input, edm::DetSetVector<SiStripProcessedRawDigi>& output) {
  for(edm::DetSetVector<SiStripRawDigi>::const_iterator DSV_it=input.begin(); DSV_it!=input.end(); DSV_it++) {
    std::vector<int16_t> int_digis(DSV_it->size());
    SiStripPedestalsSubtractor_->subtract(*DSV_it,int_digis);
    SiStripCommonModeNoiseSubtractor_->subtract(DSV_it->id,int_digis);
    std::vector<float> digis(int_digis.begin(), int_digis.end());
    common_process(digis, DSV_it->id, output);
  }
}

void 
SiStripProcessedRawDigiProducer::common_process(std::vector<float> & digis,  const uint32_t& detId, edm::DetSetVector<SiStripProcessedRawDigi>& output) {

  //Apply Gains
  SiStripApvGain::Range detGainRange =  gainHandle_->getRange(detId);   
  for(std::vector<float>::iterator it=digis.begin(); it!=digis.end(); it++) 
    (*it)/= (gainHandle_->getStripGain(it-digis.begin(), detGainRange));

  //Insert as DetSet
  edm::DetSet<SiStripProcessedRawDigi> ds(detId);
  copy(digis.begin(), digis.end(), back_inserter(ds.data) ); 
  output.insert(ds);
}
