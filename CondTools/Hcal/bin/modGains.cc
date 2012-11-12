#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <string>

#include "CalibCalorimetry/HcalAlgos/interface/HcalDbASCIIIO.h"
//#include "CondFormats/HcalObjects/interface/AllObjects.h"
#include "CondFormats/HcalObjects/interface/HcalGains.h"
#include "CondFormats/HcalObjects/interface/HcalRespCorrs.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"





int main (int argn, char* argv []) {
  if (argn < 4) {
    std::cerr << "Use: " << argv[0] << " operation <gains(.txt)> <operand> <result(.txt)>" << std::endl;
    std::cerr << "  where operation: sadd,ssub,smult,sdiv = +-*/ of a scalar value \n";
    std::cerr << "                   add,sub,mult,div = +-*/ of vector values (in RespCorr-format)\n";
    return 1;
  }
  HcalTopology topo(HcalTopologyMode::LHC,2,3);

  // get base conditions
  std::cerr << argv[2] << std::endl;
  std::ifstream inStream (argv[2]);
  HcalGains gainsIn(&topo);;
  HcalDbASCIIIO::getObject (inStream, &gainsIn);

  // where to write the result
  std::ofstream outStream (argv[4]);

  // operation and operand
  float val = 1.0;
  std::string s_operation;
  s_operation = argv[1];
  bool vectorop = false;

  HcalRespCorrs corrsIn(&topo);;

  if ( (std::strcmp(s_operation.c_str(),"add")==0) || 
       (std::strcmp(s_operation.c_str(),"sub")==0) || 
       (std::strcmp(s_operation.c_str(),"mult")==0) || 
       (std::strcmp(s_operation.c_str(),"div")==0) ) // vector operation
    {
      vectorop = true;
      std::ifstream inCorr (argv[3]);
      HcalDbASCIIIO::getObject (inCorr, &corrsIn);
    }
  else if ((std::strcmp(s_operation.c_str(),"sadd")==0) || 
	   (std::strcmp(s_operation.c_str(),"ssub")==0) || 
	   (std::strcmp(s_operation.c_str(),"smult")==0) || 
	   (std::strcmp(s_operation.c_str(),"sdiv")==0)) // scalar operation
    {
      val = atof (argv[3]);
      std::cerr << "Scalar operation: using val=" << val << std::endl;
    }
  else
    {
      std::cerr << "Unknown operator. Stopping. \n";
      return 1;
    }

  HcalGains gainsOut(&topo);;
  std::vector<DetId> channels = gainsIn.getAllChannels ();
  std::cerr << "size = " << channels.size() << std::endl;
  for (unsigned i = 0; i < channels.size(); i++) {
    DetId id = channels[i];

    if (vectorop)  // vector operation
      {
	if ((std::strcmp(s_operation.c_str(),"mult")==0)||(std::strcmp(s_operation.c_str(),"div")==0)) val = 1.0; // mult,div
	if ((std::strcmp(s_operation.c_str(),"add")==0)||(std::strcmp(s_operation.c_str(),"sub")==0)) val = 0.0; // add,sub
	if (corrsIn.exists(id))
	  {
	    val = corrsIn.getValues(id)->getValue();
	  }
	if (i%100 == 0)
	  std::cerr << "Vector operation, " << i << "th channel: using val=" << val << std::endl;
      }
    
    //    std::cerr << "val=" << val << std::endl;
    HcalGain* p_item = 0;
    if ((std::strcmp(s_operation.c_str(),"add")==0) || (std::strcmp(s_operation.c_str(),"sadd")==0))
      p_item = new HcalGain(id, gainsIn.getValues(id)->getValue(0) + val, gainsIn.getValues(id)->getValue(1) + val, 
		    gainsIn.getValues(id)->getValue(2) + val, gainsIn.getValues(id)->getValue(3) + val);

    if ((std::strcmp(s_operation.c_str(),"sub")==0) || (std::strcmp(s_operation.c_str(),"ssub")==0))
      p_item = new HcalGain(id, gainsIn.getValues(id)->getValue(0) - val, gainsIn.getValues(id)->getValue(1) - val, 
		    gainsIn.getValues(id)->getValue(2) - val, gainsIn.getValues(id)->getValue(3) - val);

    if ((std::strcmp(s_operation.c_str(),"mult")==0) || (std::strcmp(s_operation.c_str(),"smult")==0))
      p_item = new HcalGain(id, gainsIn.getValues(id)->getValue(0) * val, gainsIn.getValues(id)->getValue(1) * val, 
		    gainsIn.getValues(id)->getValue(2) * val, gainsIn.getValues(id)->getValue(3) * val);

    if ((std::strcmp(s_operation.c_str(),"div")==0) || (std::strcmp(s_operation.c_str(),"sdiv")==0))
      p_item = new HcalGain(id, gainsIn.getValues(id)->getValue(0) / val, gainsIn.getValues(id)->getValue(1) / val, 
		    gainsIn.getValues(id)->getValue(2) / val, gainsIn.getValues(id)->getValue(3) / val);


    // for all
    if (p_item)
      gainsOut.addValues(*p_item);
    //    std::cerr << i << std::endl;
  }
  // write out
  HcalDbASCIIIO::dumpObject (outStream, gainsOut);
  return 0;
}

