#include "MADParamGenerator.h"
#include "LHCOpticsApproximator.h"
#include <iostream>

int main(int argc, char *args[])  //configuration file needed
{
  if(argc<2)
  {
    std::cout<<"Configuration file needed!"<<std::endl;
    return 0;
  }
  bool generate_data = true;
  if(argc>=3)
    generate_data = (strcmp(args[2], "1")==0);

  MADParamGenerator mad_conf_gen;
  mad_conf_gen.OpenXMLConfigurationFile(args[1]);
  mad_conf_gen.MakeAllParametrizations(generate_data);

  return 0;
}
