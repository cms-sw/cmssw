#ifndef RawDataCollectorByLabel_H
#define RawDataCollectorByLabel_H


/** \class RawDataCollectorByLabel
 *
 */

#include <FWCore/Framework/interface/MakerMacros.h>
#include <FWCore/Framework/interface/EDProducer.h>
#include <DataFormats/FEDRawData/interface/FEDRawDataCollection.h> 
#include <DataFormats/Common/interface/Handle.h>
#include <FWCore/Utilities/interface/InputTag.h>

class RawDataCollectorByLabel: public edm::EDProducer {
public:
    
    ///Constructor
    RawDataCollectorByLabel(const edm::ParameterSet& pset);
    
    ///Destructor
    virtual ~RawDataCollectorByLabel();
    
    void produce(edm::Event & e, const edm::EventSetup& c); 
          
private:

    typedef std::vector<edm::InputTag>::const_iterator tag_iterator_t;

    std::vector<edm::InputTag> inputTags_ ;
    int  verbose_ ;

};

#endif
