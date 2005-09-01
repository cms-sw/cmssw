#ifndef Framework_Retriever_h
#define Framework_Retriever_h

/*----------------------------------------------------------------------
  
Retriever: The abstract interface through which the EventPrincipal
uses input services to retrieve EDProducts from external storage.

$Id: Retriever.h,v 1.2 2005/07/14 22:50:52 wmtan Exp $

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
