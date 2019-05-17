#ifndef DETECTOR_DESCRIPTION_CORE_DD_COMPACT_VIEW_H
#define DETECTOR_DESCRIPTION_CORE_DD_COMPACT_VIEW_H

class DDCompactView
{ 
public:
  
  DDCompactView(const cms::DDDetector& det)
    : m_det(det) {}

 private:
  const cms::DDDetector& m_det;
};

#endif
