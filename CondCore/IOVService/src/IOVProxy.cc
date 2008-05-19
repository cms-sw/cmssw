#include "CondCore/IOVService/interface/IOVProxy.h"
#include "CondCore/DBCommon/interface/TypedRef.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "IOV.h"



namespace cond {

  namespace impl {
    struct IOVImpl {
      cond::TypedRef<cond::IOV> iov;
    };

  }


  void IOVElement::set(IOV const & v, int i) {
    since = i==0 ? v.firstsince : v.iov[i-1].first;
    till  = v.iov[i].first;
    token = v.iov[i].second;
  }







}
