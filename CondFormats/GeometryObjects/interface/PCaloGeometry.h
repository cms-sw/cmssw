#ifndef CondFormats_PCaloGeometry_h
#define CondFormats_PCaloGeometry_h

#include <vector>
#include <stdint.h>

class PCaloGeometry{

 public:
  PCaloGeometry();
  PCaloGeometry(std::vector<double>,std::vector<double>,std::vector<uint32_t>);
  ~PCaloGeometry(){};

  std::vector<double>   getTranslation() const { return m_translation; } ;
  std::vector<double>   getDimension()   const { return m_dimension  ; } ;
  std::vector<uint32_t> getIndexes()     const { return m_indexes    ; } ;

 private:
  std::vector<double>   m_translation ;
  std::vector<double>   m_dimension   ;
  std::vector<uint32_t> m_indexes     ;

};

#endif

