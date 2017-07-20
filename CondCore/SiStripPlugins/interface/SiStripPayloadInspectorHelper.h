#ifndef CONDCORE_SISTRIPPLUGINS_SISTRIPPAYLOADINSPECTORHELPER_H
#define CONDCORE_SISTRIPPLUGINS_SISTRIPPAYLOADINSPECTORHELPER_H

#include <vector>
#include <numeric>

std::pair<float,float> getTheRange(std::map<uint32_t,float> values){
  
  float sum = std::accumulate(std::begin(values), 
			      std::end(values), 
			      0.0,
			      [] (float value, const std::map<uint32_t,float>::value_type& p)
			      { return value + p.second; }
			      );

  float m =  sum / values.size();

  float accum = 0.0;
  std::for_each (std::begin(values), 
		 std::end(values), 
		 [&](const std::map<uint32_t,float>::value_type& p) 
		 {accum += (p.second - m) * (p.second - m);}
		 );
  
  float stdev = sqrt(accum / (values.size()-1)); 

  return std::make_pair(m-2*stdev,m+2*stdev);
  
}

#endif
