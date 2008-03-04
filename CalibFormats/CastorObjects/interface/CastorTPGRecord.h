#ifndef CastorObjects_CastorTPGRecord_h
#define CastorObjects_CastorTPGRecord_h
/**\class CastorTPGRecord 

 Description: copy from HCAL

 Usage:
    <usage>

*/
//

#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"
#include "CalibFormats/CastorObjects/interface/CastorDbRecord.h"

class CastorTPGRecord : public edm::eventsetup::DependentRecordImplementation<CastorTPGRecord, boost::mpl::vector<CastorDbRecord> >{};

#endif
