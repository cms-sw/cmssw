#ifndef Fireworks_Core_FWFWEcalDetailViewBase_h
#define Fireworks_Core_FWFWEcalDetailViewBase_h

// #include "TEveViewer.h"
#include "Fireworks/Core/interface/FWDetailViewGL.h"

class TEveCaloData;
class TEveCaloLego;
class TLegend;
class FWECALDetailViewBuilder;

template <typename T>
class FWECALDetailViewBase : public FWDetailViewGL<T> {
public:
  FWECALDetailViewBase();            //: m_data(0), m_builder(0), m_legend(0) {}
  ~FWECALDetailViewBase() override;  // { delete m_data; }

protected:
  TEveCaloData *m_data;
  FWECALDetailViewBuilder *m_builder;
  TLegend *m_legend;

private:
  using FWDetailView<T>::build;
  void build(const FWModelId &id, const T *) override;

  using FWDetailViewGL<T>::setTextInfo;
  void setTextInfo(const FWModelId &id, const T *) override;
};

#include "Fireworks/Calo/src/FWECALDetailViewBase.icc"

#endif
