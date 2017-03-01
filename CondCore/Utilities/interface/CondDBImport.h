#ifndef Utilities_CondDBImport_h
#define Utilities_CondDBImport_h

#include "CondCore/CondDB/interface/Types.h"
#include "CondCore/CondDB/interface/Session.h"
#include <memory>
//

namespace cond {

  namespace persistency {

    cond::Hash import( Session& source, const cond::Hash& sourcePayloadId, const std::string& inputTypeName, const void* inputPtr, Session& destination );

    std::pair<std::string, std::shared_ptr<void> > fetch( const cond::Hash& payloadId, Session& session );
    std::pair<std::string, std::shared_ptr<void> > fetchOne( const std::string &payloadTypeName, const cond::Binary &data, const cond::Binary &streamerInfo, std::shared_ptr<void> payloadPtr );

  }

}

#endif
