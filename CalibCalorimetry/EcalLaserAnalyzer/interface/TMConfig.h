#ifndef TMConfig_H
#define TMConfig_H

#include "TObject.h"

#define fNsmNmax 36  //number of SM
#define fNlmodN 9    //number of lmodN in a SM
#define fNmem 10     //number of PNs in a MEM
#define fNmodN 4     //number of modN in a SM
#define fNtt 68      //number of trigger towers in a SM
#define fNmax 8
#define fNburmax 3
#define fNseqmax 3
#define fNcolors 6   //number of laser colors

class TMConfig: public TObject 
{

 private:
  int smin;
  int arr[fNsmNmax+1][fNmodN+1];
  int nbof[fNsmNmax+1][fNlmodN+1];
  int towerlist[fNsmNmax+1][fNlmodN][fNmax+1];
  int channlist[fNsmNmax+1][fNlmodN][fNmax+1];
  int addrpn[fNsmNmax+1][fNmodN+1][fNmem];
  int n_pin[fNsmNmax+1][fNmodN+1];
  int seqTypeOfSignal[fNseqmax+1],numbOfEventperBurstAndSignal[fNseqmax+1];
  int numbOfBurstperSignal[fNseqmax+1];
  int ped_size[fNburmax+1],laser_size[fNcolors][fNseqmax+1];
  double alpha[fNcolors],beta[fNcolors];
  double alpha_run[fNcolors][fNsmNmax+1][fNtt],beta_run[fNcolors][fNsmNmax+1][fNtt];

  void init();
  void readSequenzaConfig();
  void readlmodNConfig();
  void readpnConfig();
  void initShapeAnalysis();
  void initLaserPulseFit();
  void initTPFit();
  void initPNFit();
  void initMatacqPulseFit();

  int convert(int);

  int firstSample, lastSample;
  double alpha_start, beta_start;

  int firstpnSample, lastpnSample;
  int nbofiter,nbofpresamp,samplemin,samplemax;
  int nbofpnpresamp, nbofpnsamp, nbofsamp;

  int nbofmtqsamples,nbofmtqpresamp,vlastmtqsample,nbofmtqsigmas;
  int nbofmtqsamp1esbeforemax_parab,nbofmtqsamplesaftermax_parab;
  int thres_mtq,ampllow_trise,amplhigh_trise;

 public:
  // Default Constructor, mainly for Root
  TMConfig();

  // Destructor: Does nothing
  virtual ~TMConfig();

  int getfirstSM() {return smin;}
  int getfirstSample() {return firstSample;}
  int getlastSample() {return lastSample;}
  int getfirstPNSample() {return firstpnSample;}
  int getlastPNSample() {return lastpnSample;}
  float getalpha0() {return alpha_start;}
  float getbeta0() {return beta_start;}
  int getsampleMin() {return samplemin;}
  int getsampleMax() {return samplemax;}
  int getNbOfxtalpresamples() {return nbofpresamp;}
  int getNbOfPNpresamples() {return nbofpnpresamp;}
  int getNbOfiterations() {return nbofiter;}
  int getNbOfPNsamples() { return nbofpnsamp;}
  int getNbOfxtalsamples() { return nbofsamp;}
  double getalpha_ls(int c) { return alpha[c];}
  double getbeta_ls(int c) { return beta[c];}

  void loadPParams();
  double getalpha_run(int,int,int);
  double getbeta_run(int,int,int);

  int getNbOf(int,int);
  int getTNumb(int,int,int);
  int getXNumb(int,int,int);
  int getPNaddr(int,int,int);
  int getNbOfPNs(int,int);

  int getNbOfMatacqsamples() {return nbofmtqsamples;}
  int getNbOfMatacqpresamples() {return nbofmtqpresamp;}
  int getvlastMatacqsample() {return vlastmtqsample;}
  int getNoiseCutForMatacq() {return nbofmtqsigmas;}
  int getNbOfsamplesBefMax() {return nbofmtqsamp1esbeforemax_parab;}
  int getNbOfsamplesAftMax() {return nbofmtqsamplesaftermax_parab;}
  int getThresForMatacq() {return thres_mtq;}
  int getLowLevelForTRise() {return ampllow_trise;}
  int getHighLevelForTRise() {return amplhigh_trise;}

  int getSignalTypeForSeq(int seqNumb) { return seqTypeOfSignal[seqNumb];}
  int getNbOfBurstperSignalForSeq(int seqNumb) { return numbOfBurstperSignal[seqNumb];}
  int getNbOfEventperBurstAndSignalForSeq(int seqNumb) { return numbOfEventperBurstAndSignal[seqNumb];}

  //  ClassDef(TMConfig,1)
};

#endif

