#ifndef Framework_InputSource_h
#define Framework_InputSource_h


/*----------------------------------------------------------------------
  
InputSource: Abstract interface for all primary input sources. Input
sources are responsible for creating an EventPrincipal, using data
controlled by the source, and external to the EventPrincipal itself.

The InputSource is also responsible for dealing with the "process
name list" contained within the EventPrincipal. Each InputSource has
to know what "process" (HLT, PROD, USER, USER1, etc.) the program is
part of. The InputSource is repsonsible for pushing this process name
onto the end of the process name list.

For now, we specify this process name to the constructor of the
InputSource. This should be improved.

 Some questions about this remain:

   1. What should happen if we "rerun" a process? i.e., if "USER1" is
   already last in our input file, and we run again a job which claims
   to be "USER1", what should happen? For now, we just quietly add
   this to the history.

   2. Do we need to detect a problem with a history like:
         HLT PROD USER1 PROD
   or is it up to the user not to do something silly? Right now, there
   is no protection against such sillyness.

Some examples of InputSource subclasses may be:

 1) EmptySource: creates EventPrincipals which contain no EDProducts.
 2) PoolSource: creates EventPrincipals which "contain" the data
    read from a POOL file. This source should provide for delayed loading
    of data, thus the quotation marks around contain.
 3) DAQInputSource: creats EventPrincipals which contain raw data, as
    delivered by the L1 trigger and event builder. 

$Id: InputSource.h,v 1.2 2005/10/05 04:00:03 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <string>

namespace edm {
  class EventPrincipal;
  class ProductRegistry;
  class InputSourceDescription;
  class ModuleDescription;
  class InputSource {
  public:
    explicit InputSource(InputSourceDescription const&);
    virtual ~InputSource();

    // Indicate inability to get a new event by returning a null
    // auto_ptr.
    std::auto_ptr<EventPrincipal> readEvent();

    void addToRegistry(ModuleDescription const& md) {
      this->addToReg(md);
    }

    ProductRegistry & productRegistry() const {return *preg_;}
    
  protected:
    // A pointer to the ProductRegistry;
    ProductRegistry * preg_;

    // The process name we add to each EventPrincipal.
    std::string const process_;

  private:
    // Indicate inability to get a new event by returning a null
    // auto_ptr.
    virtual std::auto_ptr<EventPrincipal> read() = 0;

    virtual void addToReg(ModuleDescription const&);
  };
}

#endif
