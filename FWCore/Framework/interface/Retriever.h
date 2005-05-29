#ifndef EDM_RETRIEVER_HH
#define EDM_RETRIEVER_HH

/*----------------------------------------------------------------------
  
Retriever: The abstract interface through which the EventPrincipal
uses input services to retrieve EDProducts from external storage.

$Id: Retriever.h,v 1.3 2005/05/03 19:27:52 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>

#include "FWCore/CoreFramework/interface/BranchKey.h"
#include "FWCore/EDProduct/interface/EDProduct.h"


namespace edm
{
  class Retriever
  {
  public:
    virtual ~Retriever();

    virtual std::auto_ptr<EDProduct> get(BranchKey const& k) = 0;
  };
}

#endif
