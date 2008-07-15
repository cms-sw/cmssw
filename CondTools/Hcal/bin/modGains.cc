#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"


int main (int argn, char* argv []) {
  if (argn < 4) {
    std::cerr << "Use: " << argv[0] << " operation <gains(.txt)> <operand> <result(.txt)>" << std::endl;
    std::cerr << "  where operation: sadd,ssub,smult,sdiv = +-*/ of a scalar value \n";
    std::cerr << "                   add,sub,mult,div = +-*/ of vector values (in RespCorr-format) \n";
    return 1;
  }

  // get base conditions
  std::ifstream inStream (argv[2]);
  HcalGains gainsIn;
  HcalDbASCIIIO::getObject (inStream, &gainsIn);

  // where to write the result
  std::ofstream outStream (argv[4]);

  // operation and operand
  float val = 1.0;
  char s_operation[20];
  strcpy(s_operation,argv[1]);
  bool vectorop = false;

  HcalRespCorrs corrsIn;

  if ( (strcmp(s_operation,"add")==0) || 
       (strcmp(s_operation,"sub")==0) || 
       (strcmp(s_operation,"mult")==0) || 
       (strcmp(s_operation,"div")==0) ) // vector operation
    {
      vectorop = true;
      std::ifstream inCorr (argv[3]);
      HcalDbASCIIIO::getObject (inCorr, &corrsIn);
    }
  else if ((strcmp(s_operation,"sadd")==0) || 
	   (strcmp(s_operation,"ssub")==0) || 
	   (strcmp(s_operation,"smult")==0) || 
	   (strcmp(s_operation,"sdiv")==0)) // scalar operation
    val = atof (argv[3]);
  else
    {
      std::cerr << "Unknown operator. Stopping. \n";
      return 1;
    }

  HcalGains gainsOut;
  std::vector<DetId> channels = gainsIn.getAllChannels ();
  
  for (unsigned i = 0; i < channels.size(); i++) {
    DetId id = channels[i];

    if (vectorop)  // vector operation
      {
	if ((strcmp(s_operation,"mult")==0)||(strcmp(s_operation,"div")==0)) val = 1.0; // mult,div
	if ((strcmp(s_operation,"add")==0)||(strcmp(s_operation,"sub")==0)) val = 0.0; // add,sub
	if (corrsIn.exists(id))
	  {
	    val = corrsIn.getValues(id)->getValue();
	  }
      }
    
    HcalGain* p_item = 0;
    if ((strcmp(s_operation,"add")==0) || (strcmp(s_operation,"sadd")==0))
      p_item = new HcalGain(id, gainsIn.getValues(id)->getValue(0) + val, gainsIn.getValues(id)->getValue(1) + val, 
		    gainsIn.getValues(id)->getValue(2) + val, gainsIn.getValues(id)->getValue(3) + val);

    if ((strcmp(s_operation,"sub")==0) || (strcmp(s_operation,"ssub")==0))
      p_item = new HcalGain(id, gainsIn.getValues(id)->getValue(0) - val, gainsIn.getValues(id)->getValue(1) - val, 
		    gainsIn.getValues(id)->getValue(2) - val, gainsIn.getValues(id)->getValue(3) - val);

    if ((strcmp(s_operation,"mult")==0) || (strcmp(s_operation,"smult")==0))
      p_item = new HcalGain(id, gainsIn.getValues(id)->getValue(0) * val, gainsIn.getValues(id)->getValue(1) * val, 
		    gainsIn.getValues(id)->getValue(2) * val, gainsIn.getValues(id)->getValue(3) * val);

    if ((strcmp(s_operation,"div")==0) || (strcmp(s_operation,"sdiv")==0))
      p_item = new HcalGain(id, gainsIn.getValues(id)->getValue(0) / val, gainsIn.getValues(id)->getValue(1) / val, 
		    gainsIn.getValues(id)->getValue(2) / val, gainsIn.getValues(id)->getValue(3) / val);


    // for all
    if (p_item)
      gainsOut.addValues(*p_item);
  }
  // write out
  HcalDbASCIIIO::dumpObject (outStream, gainsOut);
  return 0;
}

