#ifndef HtrXmlPatternToolParameters_h_included
#define HtrXmlPatternToolParameters_h_included 1

#include <iostream>

class HtrXmlPatternToolParameters {
public:
  HtrXmlPatternToolParameters();
  ~HtrXmlPatternToolParameters();
  void Print();

  bool         m_show_errors;
  int          m_presamples_per_event;
  int          m_samples_per_event;

  int          m_XML_file_mode;
  std::string  m_file_tag;
  std::string  m_user_output_directory;
  std::string  m_output_directory;
protected:
private:
};

#endif
