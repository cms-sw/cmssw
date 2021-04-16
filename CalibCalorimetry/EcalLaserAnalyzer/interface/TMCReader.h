#ifndef TMCReader_H
#define TMCReader_H

#include "TObject.h"

class TMCReader : public TObject {
public:
  static constexpr unsigned int FNPNMAX = 10;
  static constexpr unsigned int FNLMODNMAX = 9;
  static constexpr unsigned int FNCHANMAX = 200;
  static constexpr unsigned int fNpns = 2;
  static constexpr unsigned int fNchans = 400;
  static constexpr unsigned int fNbins = 102;

private:
  int smN, nlmodN, arr[FNLMODNMAX];
  long int timestart, timestop;
  float evts[fNpns + 1][FNCHANMAX + FNPNMAX];
  double min[fNpns + 1][FNCHANMAX + FNPNMAX], max[fNpns + 1][FNCHANMAX + FNPNMAX];
  double val[fNpns + 1][FNCHANMAX + FNPNMAX], sig[fNpns + 1][FNCHANMAX + FNPNMAX];
  double wbin[fNpns + 1][FNCHANMAX + FNPNMAX];
  float sumprob;

  int smlocal, color, lmdir, part;

  void init();

public:
  // Default Constructor, mainly for Root
  TMCReader();

  // Destructor: Does nothing
  virtual ~TMCReader();

  void validMCLaser(int, int);
  void getMCLaserData(int, int);
  void validMCPulse(int);
  void getMCPulseData(int);

  int getSMNumb() { return smN; }
  int getNbOflmodN() { return nlmodN; }
  int getlmodN(int indx) { return arr[indx]; }
  int getstartime() { return timestart; }
  int getstoptime() { return timestop; }
  int getnevts(int norm) { return (int)evts[norm][0]; }

  void setsmlocal(int sm) { smlocal = sm; }
  void setcolor(int c) { color = c; }
  void setdirlmodN(int lmp) { lmdir = lmp; }
  void setpartition(int p) { part = p; }

  void changedatatoraw(int, int, int);
  void changedatatopeak(int, int, int);

  void printeinjData(int, int, int);
  void printlaserData(int, int, int, int);
  void printlaserpeak(int, int, int);
  void printnormlaserData(int, int, int, int, int);
  void printnormlaserpeak(int, int, int, int);

  //  ClassDef(TMCReader,1)
};

#endif
