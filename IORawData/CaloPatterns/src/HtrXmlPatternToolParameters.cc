#include "HtrXmlPatternToolParameters.h"
#include <iostream>

HtrXmlPatternToolParameters::HtrXmlPatternToolParameters() {}

HtrXmlPatternToolParameters::~HtrXmlPatternToolParameters() {}

void HtrXmlPatternToolParameters::Print() {
  using namespace std;
  cout << "show_errors           = " << m_show_errors << endl;
  cout << "presamples_per_event  = " << m_presamples_per_event << endl;
  cout << "samples_per_event     = " << m_samples_per_event << endl;
  cout << "XML_file_mode         = " << m_XML_file_mode << endl;
  cout << "file_tag              = " << m_file_tag << endl;
  cout << "user_output_directory = " << m_user_output_directory << endl;
  cout << "output_directory      = " << m_output_directory << endl;
}
