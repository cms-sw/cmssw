#ifndef GBRWrapper_h
#define GBRWrapper_h

#include "CondFormats/EgammaObjects/interface/GBRForest.h"

class GBRWrapper 
{

  public:
  GBRWrapper() {}
  GBRWrapper(const GBRForest &forest) : m_forest(forest) {}

  const GBRForest &GetForest() const { return m_forest; }

  private:
    GBRForest m_forest;
  
};

#endif //GBRWrapper_h
