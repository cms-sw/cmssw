#ifndef FWCore_Framework_CPCSentry_h
#define FWCore_Framework_CPCSentry_h

// class CPCSentry uses RAII to make sure that the
// CurrentProcessingContext pointer it is guarding is set to the right
// value, and cleared at the right time.

namespace edm
{
  class CurrentProcessingContext;
  namespace detail
  {

    class CPCSentry
    {
    public:
      CPCSentry(CurrentProcessingContext const*& c,
		CurrentProcessingContext const* value) :
	referenced_(&c)
      {
	c = value;	
      }

      ~CPCSentry() { *referenced_ = 0; }

    private:
      CurrentProcessingContext const** referenced_;
    };
  }
}


#endif // FWCore_Framework_CPCSentry_h
