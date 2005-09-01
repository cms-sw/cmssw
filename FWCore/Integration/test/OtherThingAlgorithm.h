#ifndef Integration_OtherThingAlgorithm_h
#define Integration_OtherThingAlgorithm_h

#include <vector>

namespace edm {
  class Event;
}
  
namespace edmreftest {

  class OtherThing;
  typedef std::vector<OtherThing> OtherThingCollection;
  
  class OtherThingAlgorithm {
  public:
    OtherThingAlgorithm() : theDebugLevel(0) {}
  
    /// Runs the algorithm and returns a list of OtherThings
    /// The user declares the vector and calls this method.
    void run(edm::Event & e, OtherThingCollection& otherThingCollection);
  
  private:
    int    theDebugLevel;
  };

}
  
#endif
