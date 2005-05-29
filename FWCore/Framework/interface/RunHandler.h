#ifndef EDM_RUN_HANDLER_INCLUDED
#define EDM_RUN_HANDLER_INCLUDED

// Base class for things that deal with begin/end run.
// Questions:
//    Should the two concepts be separated?
//    Should the function *not* be pure virtual?
//    How many other sorts of interface should there be?

namespace edm
{
  class RunHandler
    {
    public:
      virtual ~RunHandler();
      void    outerBeginRun(Run& r);
      void    outerEndRun(Run& r);
    private:
      virtual void beginRun(Run& r) = 0;
      virtual void endRun(Run& r) = 0;
    };
}

#endif //  EDM_RUN_HANDLER_INCLUDED
