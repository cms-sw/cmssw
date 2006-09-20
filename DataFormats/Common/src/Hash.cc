#include "DataFormats/Common/interface/Hash.h"

namespace edm
{
  namespace detail
  {
    // This string is the 16-byte, non-printable version.
    std::string const& InvalidHash()
    {
      static const std::string invalid = cms::MD5Result().compactForm();
      return invalid;
    }
  }
}
