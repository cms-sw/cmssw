#ifndef EDM_RETRIEVER_HH
#define EDM_RETRIEVER_HH

/*----------------------------------------------------------------------
  
Retriever: The abstract interface through which the EventPrincipal
uses input services to retrieve EDProducts from external storage.

$Id: Retriever.h,v 1.1 2005/05/29 02:29:53 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>

#include "FWCore/Framework/interface/BranchKey.h"
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
