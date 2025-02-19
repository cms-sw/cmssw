#ifndef PhysicsTFormulaPayload_h
#define PhysicsTFormulaPayload_h

#include <string>
#include <vector>

class PhysicsTFormulaPayload
{
 public:
  PhysicsTFormulaPayload(){}
  PhysicsTFormulaPayload(const std::vector< std::pair<float, float> >& l,
			   const std::vector<std::string>& f): limits_(l), formulas_(f){}
const std::vector< std::pair<float, float> >& limits() const {return limits_;}  
const std::vector<std::string>& formulas() const {return formulas_;}
 protected:
  // internally it has to contains >= 1 formula and accordingly limits
  std::vector< std::pair<float, float> > limits_;
  std::vector<std::string> formulas_;

};

#endif

