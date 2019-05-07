#ifndef CastorObjects_CastorTPGRecord_h
#define CastorObjects_CastorTPGRecord_h
/**\class CastorTPGRecord

 Description: copy from HCAL

 Usage:
    <usage>

*/
//

#include "CalibFormats/CastorObjects/interface/CastorDbRecord.h"
#include "FWCore/Framework/interface/EventSetupRecordImplementation.h"

class CastorTPGRecord
    : public edm::eventsetup::DependentRecordImplementation<
          CastorTPGRecord, boost::mpl::vector<CastorDbRecord>> {};

#endif
