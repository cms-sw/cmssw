#ifndef DQMSERVICES_CORE_STANDALONE_H
# define DQMSERVICES_CORE_STANDALONE_H
# if WITHOUT_CMS_FRAMEWORK
#  include <string>

namespace edm
{
  class ParameterSet
  {
  public:
    template <class T> static const T &
    getUntrackedParameter(const char * /* key */, const T &value)
    { return value; }
  };
  std::string getReleaseVersion(void) { return "CMSSW_STANDALONE"; }
}
# endif // WITHOUT_CMS_FRAMEWORK
#endif // DQMSERVICES_CORE_STANDALONE_H
