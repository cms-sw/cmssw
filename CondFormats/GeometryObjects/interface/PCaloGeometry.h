#ifndef CondFormats_PCaloGeometry_h
#define CondFormats_PCaloGeometry_h

#include <vector>
#include <stdint.h>

class PCaloGeometry{

 public:
  PCaloGeometry();
  PCaloGeometry(std::vector<float> const & ,
		std::vector<float> const & ,
		std::vector<uint32_t> const &,
		std::vector<uint32_t> const & );
    
  ~PCaloGeometry(){};

  std::vector<float> const &  getTranslation() const { return m_translation; }
  std::vector<float> const & getDimension() const { return m_dimension; }
  std::vector<uint32_t> const & getIndexes() const { return m_indexes; }
  std::vector<uint32_t> const & getDenseIndices() const { return m_dins; }

 private:
  std::vector<float>    m_translation ;
  std::vector<float>    m_dimension   ;
  std::vector<uint32_t> m_indexes     ;
  std::vector<uint32_t> m_dins        ;
};

#endif

