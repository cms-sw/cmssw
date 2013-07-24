#ifndef SecondaryInput_SecondaryProducer_h
#define SecondaryInput_SecondaryProducer_h

/** \class SecondaryProducer
 *
 * \author Bill Tanenbaum
 *
 *
 ************************************************************/

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Sources/interface/VectorInputSource.h"
#include "IORawData/DaqSource/interface/DaqBaseReader.h"

#include <string>

namespace evf {

  class SecondaryProducer: public DaqBaseReader
  {
  public:
    //
    // construction/destruction
    //
    explicit SecondaryProducer(const edm::ParameterSet& ps);
    virtual ~SecondaryProducer();
    
    
    //
    // public member functions
    //
    
    // DaqBaseReader interface
    virtual int fillRawData(edm::EventID& eID,
			    edm::Timestamp& tstamp, 
			    FEDRawDataCollection*& data);
    
    void processOneEvent(edm::EventPrincipal const& eventPrincipal, edm::EventID& eID, FEDRawDataCollection*& data);
    
  private:
    //
    // private member functions
    //
    boost::shared_ptr<edm::VectorInputSource> makeSecInput(edm::ParameterSet const& ps);


  private:
    //
    // member data
    //
    edm::RunNumber_t   runNum;
    edm::EventNumber_t eventNum;

    boost::shared_ptr<edm::VectorInputSource> secInput_;

    std::string moduleLabel_;
    std::string instance_;
    std::string process_;
    size_t cachedOffset_;
    int fillCount_;
  };

} // evf


#endif
