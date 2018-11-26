#ifndef CondFormats_SiPixelObjects_SiPixelQualityProbabilities_h 
#define CondFormats_SiPixelObjects_SiPixelQualityProbabilities_h

#include "CondFormats/Serialization/interface/Serializable.h"

#include <map>
#include <string>
#include <vector>

class SiPixelQualityProbabilities{
public:
  typedef std::vector<std::pair<std::string,float> > probabilityVec; 
  typedef std::map<unsigned int,probabilityVec> probabilityMap; 
  
  SiPixelQualityProbabilities(){}
  virtual ~SiPixelQualityProbabilities(){}

  void setProbabilities(const unsigned int &puBin, const probabilityVec &theProbabilities);
              
  const probabilityMap& getProbability_Map () const  {return m_probabilities;}
   
  probabilityVec   getProbabilities(const unsigned int &puBin) const;
  probabilityVec & getProbabilities(const unsigned int &puBin);
  
  double size()const {return m_probabilities.size();}
  double nelements(const int &puBin)const {return m_probabilities.at(puBin).size();}
  std::vector<unsigned int> getPileUpBins() const;

  void printAll() const;

private:

  probabilityMap m_probabilities;

  COND_SERIALIZABLE;

};

#endif
