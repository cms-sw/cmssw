#ifndef EDM_POOLINPUTSERVICE_H
#define EDM_YPOOLINPUTSERVICE_H

/*----------------------------------------------------------------------

Event streaming input service

$Id$

----------------------------------------------------------------------*/

#include <vector>
#include <memory>
#include <string>
#include <fstream>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/InputService.h"
#include "FWCore/Framework/interface/Retriever.h"
#include "FWCore/EDProduct/interface/EDProduct.h"
#include "FWCore/Framework/interface/EventPrincipal.h"
#include "FWCore/Framework/interface/BranchKey.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/EventProvenance.h"
#include "FWCore/Framework/interface/EventAux.h"

class TClass;

namespace edm {

  class EventStreamInput : public InputService
  {
    //------------------------------------------------------------
    // Nested class PoolRetriever: pretends to support file reading.
    //
    class StreamRetriever : public Retriever
    {
    public:
      virtual ~StreamRetriever();
      virtual std::auto_ptr<EDProduct> get(BranchKey const& k);
    };

    //------------------------------------------------------------

  public:
    explicit EventStreamInput(ParameterSet const& pset,
			      InputServiceDescription const& desc);
    virtual ~EventStreamInput();

  private:
    typedef std::vector<char> Buffer;

    int buffer_size_;
    Buffer event_buffer_;

    std::string const file_;
    std::ifstream ist_;
    StreamRetriever store_;
    TClass* send_event_;

    virtual std::auto_ptr<EventPrincipal> read();
    void init();

    // EventAux not handled
  };
}
#endif
