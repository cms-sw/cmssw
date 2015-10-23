#ifndef __GMTInputCaloSum_h
#define __GMTInputCaloSum_h

#include "DataFormats/L1Trigger/interface/BXVector.h"

namespace l1t {
  class GMTInputCaloSum;
  typedef BXVector<GMTInputCaloSum> GMTInputCaloSumBxCollection;

  class GMTInputCaloSum {
    public:
      GMTInputCaloSum() :
        m_etBits(0), m_hwPhi(0), m_hwEta(0), m_index(0) {};

      GMTInputCaloSum(int pt, int phi, int eta, int index) :
        m_etBits(pt), m_hwPhi(phi), m_hwEta(eta), m_index(index) {};

      virtual ~GMTInputCaloSum() {};

      void setEtBits(int bits) { m_etBits = bits; };
      void setPhiBits(int bits) { m_hwPhi = bits; };
      void setEtaBits(int bits) { m_hwEta = bits; };
      void setIndex(int idx) { m_index = idx; };

      const int etBits() const { return m_etBits; };
      const int hwPhi() const { return m_hwPhi; };
      const int hwEta() const { return m_hwEta; };
      const int index() const { return m_index; };

    private:
      int m_etBits;
      int m_hwPhi;
      int m_hwEta;
      int m_index;
  };
}

#endif
