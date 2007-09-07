#ifndef FastL1_h
#define FastL1_h

#include <vector>

// Defining my own  class
class FastL1BitInfo
{

public:

FastL1BitInfo(): eta(0), phi(0), energy(0), et(0), TauVeto(false), EmTauVeto(false), HadTauVeto(false), IsolationVeto(false), SumEtBelowThres(false), maxEt(false), soft(false)
{
}

~FastL1BitInfo()
{
}

void setEta(double Eta){eta = Eta;}
void setPhi(double Phi){phi = Phi;}
void setEnergy(double Energy){energy = Energy;}
void setEt(double Et){et = Et;}
void setTauVeto(bool tauVeto){TauVeto = tauVeto;}
void setEmTauVeto(bool emTauVeto){EmTauVeto = emTauVeto;}
void setHadTauVeto(bool hadTauVeto){HadTauVeto = hadTauVeto;}
void setIsolationVeto(bool isolationVeto){IsolationVeto = isolationVeto;}
void setSumEtBelowThres(bool sumEtBelowThres){SumEtBelowThres = sumEtBelowThres;}
void setMaxEt(bool MaxEt){maxEt = MaxEt;}
void setSoft(bool Soft){soft = Soft;}

double getEta() const {return eta;}
double getPhi() const {return phi;}
double getEnergy() const {return energy;}
double getEt() const {return et;}
bool getTauVeto() const {return TauVeto;}
bool getEmTauVeto() const {return EmTauVeto;}
bool getHadTauVeto() const {return HadTauVeto;}
bool getIsolationVeto() const {return IsolationVeto;}
bool getSumEtBelowThres() const {return SumEtBelowThres;}
bool getMaxEt() const {return maxEt;}
bool getSoft() const {return soft;}

//  bool softVeto;
//  int  emEtaPattern;
//  int  hadEtaPattern;
//  int  emPhiPattern;
//  int  hadPhiPattern;
//  int  Window[8];
//std::vector< double> ecal;
//std::vector< double> hcal;
//double ecal[16];
//double hcal[16];

private:
  double eta;
  double phi;
  double energy;
  double et;

  bool TauVeto;
  bool EmTauVeto;
  bool HadTauVeto;
  bool IsolationVeto;
  bool SumEtBelowThres;
  bool maxEt;
  bool soft;

};

// Defining vector of my classs
typedef std::vector<FastL1BitInfo> FastL1BitInfoCollection;

#endif
