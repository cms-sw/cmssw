#ifndef TCalibData_H
#define TCalibData_H

#include <string>
#include <map>
#include<vector>

#define NCRYS    1700 
#define NPN        10 
#define NPARLIN     4 
#define NTAUS       2 
#define NGAIN       2 
#define NMEMEE      4
using namespace std;

class TFile;
class TH2D;
class TTree;
class TBranch;
class TCalibData
{

 private:	
  bool _debug;
  int _fed;
  string _path;
  string _sprAPDfile;
  string _sprPNfile;
  string _ABfile;
  string _linPNfile;
  string _EENumfile;
  bool   _newLinType;

  unsigned int  _nmemEE;
  bool _tauRead;
  bool _linPNRead;
  map < int, bool> _ABRead;
  bool _ABFileSet;
  bool _matacqRead;
  bool _isBarrel;

  double _tausAPD[NTAUS][NCRYS];
  double _tausPNEB[NTAUS][NPN];
  double _tausPNEE[NTAUS][NPN][NMEMEE];
  double _linPNEB[NGAIN][NPARLIN][NPN];
  double _linPNEE[NGAIN][NPARLIN][NPN][NMEMEE];
 
  double _qmaxPNEB[NPN];
  double _qmaxPNEE[NPN][NMEMEE];



  TFile  * EENumFile;
  TH2D   * EENum;
  TTree  * ABTree[4];
  TFile  * ABFile;
 
  int channelAB, ietaAB, iphiAB;
  double alpha, beta;
  TBranch        *b_channel;   //!
  TBranch        *b_iphi;   //!
  TBranch        *b_ieta;   //!
  TBranch        *b_alpha;   //!
  TBranch        *b_beta;   //!

 public:
  // Default Constructor, mainly for Root
  TCalibData();

  // Default Constructor, mainly for Root
  TCalibData(int, string);
  TCalibData(int , string  , string );

  // Default Constructor, mainly for Root
  void init(int, string);

  // Destructor: Does nothing
  virtual ~TCalibData();
  
  
  bool readTaus();
  bool readLinPN();
  bool readAB(int);
  double getqmax( double , double );
  // bool readMat();

  bool setABFile(string alphafile);
  vector<double> linPN( int, int ); 
  vector<double> linPN( int, int, int ); 

  pair<double,double> tauAPD( int );
  pair<double,double> tauAPD( int, int );
  pair<double,double> tauPN( int ); 
  pair<double,double> tauPN( int, int ); 

  pair < double, double> AB( int eta, int phi, int color);
  pair < double, double> AB( int chan, int color, int& ieta, int& iphi);

  double qmaxPN(  int ); 
  double qmaxPN( int, int ); 
  double getPNCorrected( double val0 , int iPN, int gain , int imem);

  int PNOffset( int );
  int EEchannel(int , int );


  //  ClassDef(TCalibData,1)
};

#endif



