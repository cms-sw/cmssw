#ifndef DPGAnalysis_SiStripTools_Multiplicities_H
#define DPGAnalysis_SiStripTools_Multiplicities_H

namespace sistriptools::values {
  class Multiplicity {
  public:
    explicit Multiplicity(int iMult) : m_mult{iMult} {}
    int mult() const { return m_mult; }

  private:
    int m_mult;
  };

  template <class T1, class T2>
  class MultiplicityPair {
  public:
    MultiplicityPair(T1 const& i1, T2 const& i2) : m_multiplicity1(i1), m_multiplicity2(i2) {}

    int mult1() const;
    int mult2() const;

  private:
    T1 m_multiplicity1;
    T2 m_multiplicity2;
  };

  template <class T1, class T2>
  int MultiplicityPair<T1, T2>::mult1() const {
    return m_multiplicity1.mult();
  }

  template <class T1, class T2>
  int MultiplicityPair<T1, T2>::mult2() const {
    return m_multiplicity2.mult();
  }
}  // namespace sistriptools::values
#endif  // DPGAnalysis_SiStripTools_Multiplicities_H
