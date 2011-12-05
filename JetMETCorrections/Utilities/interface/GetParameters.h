#ifndef GetParameters_h
#define GetParameters_h

#include <map>
#include <string>
#include <vector>


class KtParam{
public:
KtParam(int ktAngle, int ktRParam): theKtAngle(ktAngle), theKtRParam(ktRParam){}
~KtParam(){}
int getKtAngle(){return theKtAngle;}
int getKtRParam(){return theKtRParam;}
private:
int theKtAngle;
int theKtRParam;
};


class JetParameters{
public:
JetParameters(){}
~JetParameters(){}
void setParameters(std::string alg, int recom, int it, std::string threshold)
{
  theAlgType = alg; 
  if(recom == 1) {theRecomSchema = "EScheme";} else {theRecomSchema = "EtScheme";};
  if (it == 0) {iTime = "Jets873_2x1033PU_qcd";} else {iTime = "None";}
  theThreshold = threshold;
}

 std::string getAlgType(){return theAlgType;}
 std::string getRecomSchema(){return theRecomSchema;}
 std::string getTime(){return iTime;}
 std::string getThreshold(){return theThreshold;}

private:
std::string theAlgType;
std::string theRecomSchema;
std::string iTime;
std::string theThreshold;
};


class GetParameters{

 public:
  
  GetParameters(JetParameters);
  ~GetParameters(){};
  JetParameters get(){return theJetPar;};

 private:
 JetParameters theJetPar;
};

#endif
