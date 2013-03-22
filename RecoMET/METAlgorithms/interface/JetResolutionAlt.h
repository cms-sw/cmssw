// -*- C++ -*-
// $Id: METProducer.h,v 1.31 2013/03/06 19:31:56 vadler Exp $

#ifndef JETRESOLUTIONALT_H
#define JETRESOLUTIONALT_H

//____________________________________________________________________________||
#include <string>
#include <vector>

//____________________________________________________________________________||
class TF1;
class JetCorrectorParameters;

//____________________________________________________________________________||
class JetResolutionAlt
{
public:

  JetResolutionAlt(const std::string& fileName,bool doGaussian = false);
  virtual ~JetResolutionAlt();

  double parameterEtaEval(const std::string& parameterName,float eta, float pt);

  
private:
  void initialize(const std::string& fileName, bool doGaussian = false);

  std::string mkResolutionNameFrom(const std::string& fileName);
  TF1* mkResolutionFunction(bool doGaussian, const std::string& formulaName, const std::string& resolutionName);

  std::vector<std::string> readLevelNames(const JetCorrectorParameters& resolutionPars);

  std::vector<TF1*> parameterFncs_;
  std::vector<JetCorrectorParameters*> parameters_;
  
};

//____________________________________________________________________________||
#endif // JETRESOLUTIONALT_H
