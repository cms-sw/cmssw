/*----------------------------------------------------------------------
$Id: EmptyInputService.h,v 1.2 2005/06/07 23:47:36 wmtan Exp $
----------------------------------------------------------------------*/

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Retriever.h"
#include "FWCore/Framework/interface/InputService.h"
#include "FWCore/EDProduct/interface/CollisionID.h"

namespace edm {
  class ParameterSet;

  class EmptyInputService : public InputService {
  public:
    explicit EmptyInputService(ParameterSet const&, const InputServiceDescription&);
    ~EmptyInputService();
  private:
    std::auto_ptr<EventPrincipal> read();
    
    CollisionID nextID_;
    int remainingEvents_;
    Retriever* retriever_;
  };


  struct FakeRetriever : public Retriever {
    virtual ~FakeRetriever();
    virtual std::auto_ptr<EDProduct> get(BranchKey const& k);
  };
}
