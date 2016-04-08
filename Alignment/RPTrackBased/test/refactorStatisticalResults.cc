/****************************************************************************
*
* This is a part of TOTEM offline software.
* Authors: 
*  Jan Kašpar (jan.kaspar@gmail.com) 
*
****************************************************************************/

#include <cstdio>
#include <string>
#include <vector>

#include "Alignment/RPDataFormats/interface/RPAlignmentCorrections.h"
#include "Alignment/RPTrackBased/interface/AlignmentGeometry.h"

using namespace std;

//----------------------------------------------------------------------------------------------------

/**
 * Putting `TotemAlignment/RPTrackBased' into BuildFile leads to a segfault in the end of the
 * program. This is a by-pass.
 **/

void AlignmentGeometry_LoadFromFile(AlignmentGeometry &ag, const std::string filename)
{
  ag.clear();

  FILE *f = fopen(filename.c_str(), "r");
  if (!f)
    throw "Cannot open file.";

  while (!feof(f)) {
    unsigned int id;
    float x, y, z, dx, dy;

    int res = fscanf(f, "%u%E%E%E%E%E", &id, &x, &y, &z, &dx, &dy);

    if (res == 6) {
      unsigned int rpNum = (id / 10) % 10;
      unsigned int detNum = id % 10;
      bool isU = (detNum % 2 != 0);
      if (rpNum == 2 || rpNum == 3)
        isU = !isU;
      ag.Insert(id, DetGeometry(z, dx, dy, x, y, isU));      
    } else 
      if (!feof(f))
        throw "Wrong format.";
  }

  fclose(f);
}

//----------------------------------------------------------------------------------------------------

int main()
{
  try {
    AlignmentGeometry geom;
    
    //geom.LoadFromFile("geometry");
    AlignmentGeometry_LoadFromFile(geom, "geometry");
  
  
    vector<string> algorithms;
    algorithms.push_back("Jan");
    algorithms.push_back("Ideal");
  
    for (vector<string>::iterator it = algorithms.begin(); it != algorithms.end(); ++it) {
      printf("> %s\n", it->c_str());
      string input_file = string("./cumulative_results_") + *it + ".xml";
      RPAlignmentCorrections input(input_file);
  
      RPAlignmentCorrections expanded, factored;
  
      input.FactorRPFromSensorCorrections(expanded, factored, geom, 2);
    
      string exp_output_file = string("expanded_results_") + *it + ".xml";
      expanded.WriteXMLFile(exp_output_file);
  
      string refac_output_file = string("refactored_results_") + *it + ".xml";
      factored.WriteXMLFile(refac_output_file);
    }
  }
  catch (...) {
    printf("Exception caught.\n");
    return 1;
  }

  return 0;
}

