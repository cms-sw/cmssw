#ifndef FastL1_h
#define FastL1_h

#include <vector>

// Defining my own  class
class FastL1BitInfo
{

public:

FastL1BitInfo(): m_eta(0), m_phi(0), m_energy(0), m_et(0), m_TauVeto(false), m_EmTauVeto(false), m_HadTauVeto(false), m_IsolationVeto(false), m_SumEtBelowThres(false), m_maxEt(false), m_soft(false), m_hard(false)
{
}

~FastL1BitInfo()
{
}

void setEta(double Eta){m_eta = Eta;}
void setPhi(double Phi){m_phi = Phi;}
void setEnergy(double Energy){m_energy = Energy;}
void setEt(double Et){m_et = Et;}
void setTauVeto(bool tauVeto){m_TauVeto = tauVeto;}
void setEmTauVeto(bool emTauVeto){m_EmTauVeto = emTauVeto;}
void setHadTauVeto(bool hadTauVeto){m_HadTauVeto = hadTauVeto;}
void setIsolationVeto(bool isolationVeto){m_IsolationVeto = isolationVeto;}
void setSumEtBelowThres(bool sumEtBelowThres){m_SumEtBelowThres = sumEtBelowThres;}
void setMaxEt(bool MaxEt){m_maxEt = MaxEt;}
void setSoft(bool Soft){m_soft = Soft;}
void setHard(bool Hard){m_hard = Hard;}

void setHighestEtTowerID (int id) {m_HighestEtTowerID = id;}
void setHighestEmEtTowerID (int id) {m_HighestEmEtTowerID = id;}
void setHighestHadEtTowerID (int id) {m_HighestHadEtTowerID = id;}

int getHighestEtTowerID () {return m_HighestEtTowerID;}
int getHighestEmEtTowerID () {return m_HighestEmEtTowerID;}
int getHighestHadEtTowerID () {return m_HighestHadEtTowerID;}

double getEta() const {return m_eta;}
double getPhi() const {return m_phi;}
double getEnergy() const {return m_energy;}
double getEt() const {return m_et;}
bool getTauVeto() const {return m_TauVeto;}
bool getEmTauVeto() const {return m_EmTauVeto;}
bool getHadTauVeto() const {return m_HadTauVeto;}
bool getIsolationVeto() const {return m_IsolationVeto;}
bool getSumEtBelowThres() const {return m_SumEtBelowThres;}
bool getMaxEt() const {return m_maxEt;}
bool getSoft() const {return m_soft;}
bool getHard() const {return m_hard;}

private:
  double m_eta;
  double m_phi;
  double m_energy;
  double m_et;

// ID is defined like this:
//  0  1  2  3
//  4  5  6  7
//  8  9 10 11
// 12 13 14 15
  int m_HighestEtTowerID;
  int m_HighestEmEtTowerID;
  int m_HighestHadEtTowerID;

  bool m_TauVeto;
  bool m_EmTauVeto;
  bool m_HadTauVeto;
  bool m_IsolationVeto;
  bool m_SumEtBelowThres;
  bool m_maxEt;
  bool m_soft;
  bool m_hard;
};

// Defining vector of my classs
typedef std::vector<FastL1BitInfo> FastL1BitInfoCollection;

#endif
