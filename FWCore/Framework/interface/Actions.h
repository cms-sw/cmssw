#ifndef EDM_ACTIONCODES_HH
#define EDM_ACTIONCODES_HH

namespace edm
{
  namespace actions
  {
    enum ActionCodes
      {
	IgnoreComletely=0,
	Rethrow,
	SkipEvent,
	FailModule,
	FailPath
      };
  }
}
#endif
