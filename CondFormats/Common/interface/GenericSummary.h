#ifndef Cond_GenericSummary_h
#define Cond_GenericSummary_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include "CondFormats/Common/interface/Summary.h"

namespace cond {

  /** Short summary of condition payoad
  */
  class GenericSummary : public Summary {
  public:
    GenericSummary();
    ~GenericSummary() override;

    //
    explicit GenericSummary(std::string const& s);

    // short message (just content to be used in a table)
    void shortMessage(std::ostream& os) const override;

    // long message (to be used in pop-up, single views)
    void longMessage(std::ostream& os) const override;

  private:
    std::string m_me;

    COND_SERIALIZABLE;
  };

}  // namespace cond

#endif
