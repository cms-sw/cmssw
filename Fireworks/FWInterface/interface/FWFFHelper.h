#ifndef Fireworks_FWInterface_FWFFHelper_h
#define Fireworks_FWInterface_FWFFHelper_h

namespace edm
{
   class ParameterSet;
   class ActivityRegistry;
}

class TRint;

class FWFFHelper
{
public:
   FWFFHelper(const edm::ParameterSet &, const edm::ActivityRegistry &);
   TRint *app() { return m_Rint; }
private:
   TRint        *m_Rint;
};

#endif
