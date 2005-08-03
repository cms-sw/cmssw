#ifndef FakeRetriever_H
#define FakeRetriever_H

/** \class FakeRetriever
 *
 *  Just fails gracefully.
 *
 *  $Date: 2005/07/13 09:09:15 $
 *  $Revision: 1.1 $
 */

#include <FWCore/Framework/interface/Retriever.h>

namespace edm
{
  class FakeRetriever : public Retriever {
  public:
    virtual ~FakeRetriever(){}
    virtual std::auto_ptr<EDProduct> get(BranchKey const& k){
      throw std::runtime_error("FakeRetriever::get called");
      return std::auto_ptr<EDProduct>(0);
    }
  };
}

#endif
