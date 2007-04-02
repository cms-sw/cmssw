#include "L1TriggerConfig/DTTPGConfigProducers/src/DTConfigTrivialProducer.h"

using std::cout;
using std::endl;
using std::vector;
using std::auto_ptr;


//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
DTConfigTrivialProducer::DTConfigTrivialProducer(const edm::ParameterSet& ps)
{
 
  cout << "Constructing an DTConfigTrivialProducer" << endl;

  //the following line is needed to tell the framework what
  // data is being produced
  setWhatProduced(this, dependsOn(&DTConfigTrivialProducer::getDTGeom));

  //now do what ever other initialization is needed
  
  //get and store parameter set 
  m_ps = ps;
}


DTConfigTrivialProducer::~DTConfigTrivialProducer()
{
 
   // do anything here that needs to be done at destruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
std::auto_ptr<DTConfigManager> DTConfigTrivialProducer::produce (const DTConfigManagerRcd& iRecord)
{
   using namespace edm::es;

   edm::ParameterSet config = m_ps.getParameter<edm::ParameterSet>("DTTPGParameters");
   std::auto_ptr<DTConfigManager> dtConfig = std::auto_ptr<DTConfigManager>( new DTConfigManager(config,m_geom) );

   return dtConfig ;
}

void DTConfigTrivialProducer::getDTGeom(const MuonGeometryRecord& iRecord)
{

  iRecord.get(m_geom);

}
