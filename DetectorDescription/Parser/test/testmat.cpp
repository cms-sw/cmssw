#include <string>
#include <vector>
#include <iostream>
#include "DDD/DDParser/interface/DDLElementaryMaterial.h"
#include "DDD/DDCore/interface/DDException.h"

int main(void)

{
  cout << "Create DDLElementaryMaterial m" << endl;
  DDLElementaryMaterial m;
  //  <ElementaryMaterial name="Carbon" density="2.265*g/cm3" symbol=" " atomicWeight="12.011*g/mole" atomicNumber="6"/>

  cout << "Initialize names" << endl;
  vector<string> names;
  names.push_back("name");
  names.push_back("density");
  names.push_back("atomicWeight");
  names.push_back("atomicNumber");

  cout << "Initialize values" << endl;
  vector<string> values;
  values.push_back("Carbon");
  values.push_back("2.265*g/cm3");
  values.push_back("12.011*g/mole");
  values.push_back("6");

  cout << "Initialize element name and namespace" << endl;
  string element = "ElementaryMaterial";
  string nmspc = "test";

  cout << "Load Attributes " << endl;
  m.loadAttributes(element, names, values, nmspc);

  cout << "Process Element " << endl;
  try {
    m.processElement(element, nmspc);
  }
  catch (DDException & e)
    {
      cout << "OOPS!  DDException: " << e << endl;
    }
  cout << "Done!!!" << endl;

}
