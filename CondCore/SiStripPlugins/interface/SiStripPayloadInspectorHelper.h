#ifndef CONDCORE_SISTRIPPLUGINS_SISTRIPPAYLOADINSPECTORHELPER_H
#define CONDCORE_SISTRIPPLUGINS_SISTRIPPAYLOADINSPECTORHELPER_H

#include <vector>

std::pair<float,float> getTheRange(std::map<uint32_t,float> values){
  
  std::vector<float> v;
  for(const auto& element : values ){
    v.push_back(element.second);
  }

  float sum = std::accumulate(std::begin(v), std::end(v), 0.0);
  float m =  sum / v.size();

  float accum = 0.0;
  std::for_each (std::begin(v), std::end(v), [&](const float d) {
      accum += (d - m) * (d - m);
    });
  
  float stdev = sqrt(accum / (v.size()-1));

  return std::make_pair(m-2*stdev,m+2*stdev);
  
}

#endif
