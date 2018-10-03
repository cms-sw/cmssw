#ifndef RawDataMapperByLabel_H
#define RawDataMapperByLabel_H


/** \class RawDataMapperByLabel
 *
 */

#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "DataFormats/FEDRawData/interface/FEDRawDataCollection.h" 
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/InputTag.h"

class RawDataMapperByLabel: public edm::stream::EDProducer<> {
public:
    
    ///Constructor
    RawDataMapperByLabel(const edm::ParameterSet& pset);
    
    ///Destructor
    ~RawDataMapperByLabel() override;
    
    void produce(edm::Event & e, const edm::EventSetup& c) override; 
          
private:

    typedef std::vector<edm::InputTag>::const_iterator tag_iterator_t;
    typedef std::vector<edm::EDGetTokenT<FEDRawDataCollection> >::const_iterator tok_iterator_t;

    std::vector<edm::InputTag> inputTags_ ;
    edm::InputTag mainCollectionTag_ ;
    std::vector<edm::EDGetTokenT<FEDRawDataCollection> > inputTokens_;
    int  verbose_ ;

};

#endif
