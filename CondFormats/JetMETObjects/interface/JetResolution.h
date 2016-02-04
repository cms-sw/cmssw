#ifndef JETRESOLUTION_H
#define JETRESOLUTION_H

#include <string>
#include <vector>

#include <TF1.h>


class JetCorrectorParameters;


class JetResolution
{
  //
  // construction / destruction
  //
public:
  JetResolution();
  JetResolution(const std::string& fileName,bool doGaussian=false);
  virtual ~JetResolution();
  

  //
  // member functions
  //
public:
  void initialize(const std::string& fileName,bool doGaussian=false);
  
  const std::string& name() const { return name_; }
  
  TF1* resolutionEtaPt(float eta,float pt) const;
  TF1* resolution(const std::vector<float>&x, const std::vector<float>&y) const;
  
  TF1* parameterEta(const std::string& parameterName,float eta);
  TF1* parameter(const std::string& parameterName,const std::vector<float>&x);
  
  const JetCorrectorParameters& parameters(int i) const { return *(parameters_[i]); }
  
  
  //
  // data members
  //
private:
  std::string                          name_;
  mutable TF1*                         resolutionFnc_;
  std::vector<TF1*>                    parameterFncs_;
  std::vector<JetCorrectorParameters*> parameters_;
  
};


#endif
