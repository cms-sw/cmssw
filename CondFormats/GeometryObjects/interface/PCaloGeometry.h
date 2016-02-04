#ifndef CondFormats_PCaloGeometry_h
#define CondFormats_PCaloGeometry_h

#include <vector>
#include <stdint.h>

class PCaloGeometry{

 public:
  PCaloGeometry();
  PCaloGeometry(std::vector<double> const & ,
		std::vector<double> const & ,
		std::vector<uint32_t> const &);
  ~PCaloGeometry(){};

  std::vector<double> const &  getTranslation() const { return m_translation; }
  std::vector<double> const & getDimension() const { return m_dimension; }
  std::vector<uint32_t> const & getIndexes() const { return m_indexes; }

 private:
  std::vector<double>   m_translation ;
  std::vector<double>   m_dimension   ;
  std::vector<uint32_t> m_indexes     ;

};

#endif

