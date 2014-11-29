#include <iostream>
#include "DetectorDescription/Core/interface/DDNumeric.h"
#include "DetectorDescription/Core/interface/DDString.h"
#include "DetectorDescription/Core/interface/DDVector.h"
#include "DetectorDescription/Core/interface/DDMap.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;

double get_value(const DDMap & m, const string & k)
{
  return m[k];
}

int main() {
  cout << "DDD Named Types, Test" << endl;
  // names of the instances to be created
  DDName numeric_name("Numeric","Namespace"),
    string_name("String","Namespace"),
    vector_name("Vector","Namespace"),
    map_name("Map","Namespace");
  // DDNumeric test
  cout << "DDNumeric:" << endl;
  DDNumeric dd_numeric(numeric_name, new double(1.1));
  cout << dd_numeric << endl;

  DDString dd_string(string_name, new string("foo-bar"));
  cout << dd_string << endl;

  dd_map_type  m;// = new dd_map_type;
  m["first"] = 1.1;
  m["second"] = 0.9;
  m["third"] = 0.001;
  DDMap dd_map(map_name,&m);
  cout << dd_map << endl;
  dd_map_type::iterator it(m.begin()),ed(m.end());
  for(; it!=ed; ++it) {
    cout << it->first << ':' << it->second << ' ';
  }
  cout << endl;
  try {
    cout << "dd_map['second']=" << get_value(dd_map,"second") << endl;
    cout << "dd_map['xyz']=" << get_value(dd_map,"xyz") << endl;
  }
  catch(const cms::Exception & e) {
    cout << "EXCEPTION: " << e.what() << endl;
  }

  vector<double> vec;
  vec.push_back(1.);
  vec.push_back(2.);
  vec.push_back(3.);
  DDVector dd_vector(vector_name,&vec);
  cout << dd_vector << endl;
  cout << dd_vector[0] << ' ' << dd_vector[2] << endl;
  
  return 0;
}
