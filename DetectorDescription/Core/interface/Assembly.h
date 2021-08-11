#ifndef DDD_DDI_ASSEMBLY_H
#define DDD_DDI_ASSEMBLY_H

#include <iostream>
#include "Solid.h"

namespace DDI {

  class Assembly : public Solid {
  public:
    Assembly();

    double volume() const override { return -1; }

    void stream(std::ostream& os) const override;
  };
}  // namespace DDI

#endif
