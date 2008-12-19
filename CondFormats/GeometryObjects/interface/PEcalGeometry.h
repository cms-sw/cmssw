#ifndef CondFormats_PEcalGeometry_h
#define CondFormats_PEcalGeometry_h

#include <vector>

class PEcalGeometry{

 public:
  PEcalGeometry();
  PEcalGeometry(std::vector<double>,std::vector<double>,std::vector<uint32_t>);
  ~PEcalGeometry(){};

  std::vector<double> getTranslation() const {return m_translation;};
  std::vector<double> getDimension() const {return m_dimension;};
  std::vector<uint32_t> getIndexes() const {return m_indexes;};

 private:
  std::vector<double> m_translation;
  std::vector<double> m_dimension;
  std::vector<uint32_t> m_indexes;

};

#endif

