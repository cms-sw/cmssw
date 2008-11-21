#ifndef CondFormats_L1TObjects_EcalTPParameters_H
#define CondFormats_L1TObjects_EcalTPParameters_H
/**
 * Author: P.Paganini, Ursula Berthon
 * Created: 20 March 2007
 * $Id: EcalTPParameters.h,v 1.4 2007/04/25 17:30:42 uberthon Exp $
 **/

#include <vector>
#include <map>

class EcalTPParameters {
 public:
   
  EcalTPParameters();
  ~EcalTPParameters();

  // getters
  std::vector<unsigned int> const * getTowerParameters(int SM, int towerInSM, bool print = false) const ;
  std::vector<unsigned int> const * getStripParameters(int SM, int towerInSM, int stripInTower, bool print = false) const ;
  std::vector<unsigned int> const * getXtalParameters(int SM, int towerInSM, int stripInTower, int xtalInStrip, bool print = false)  const;

  double getEtSatEB() const {return EtSatEB_;}
  double getEtSatEE() const {return EtSatEE_;}
  double getTPGinGeVEB(unsigned int nrSM, unsigned int nrTowerInSM, unsigned int compressedEt) const;
  double getTPGinGeVEE(unsigned int nrSM, unsigned int nrTowerInSM, unsigned int compressedEt) const;

  // setters
  void setTowerParameters(int SM, int towerInSM, std::vector<unsigned int> params) 	    
    {towerParam_[getIndex(SM,towerInSM)]=params;
    }         
  void setStripParameters(int SM, int towerInSM, int stripInTower, std::vector<unsigned int> params)  
    {stripParam_[getIndex(SM,towerInSM,stripInTower)]=params;}
  void setXtalParameters(int SM, int towerInSM, int stripInTower, int xtalInStrip, std::vector<unsigned int> params) 
    {xtalParam_[getIndex(SM,towerInSM,stripInTower,xtalInStrip)]=params;
    }
  void setPhysicsParameters(std::vector<float> params)      
    {xtalLsbEB_ = params[0]; EtSatEB_ = params[1]; ttfLowEB_ = params[2]; ttfHighEB_ = params[3]; }         
  void setConstants(const int nbMaxTowers, const int nbMaxStrips, const int nbMaxXtals, const int nrMinTccEB, const int nrMaxTccEB);
  void changeThresholds(double ttfLowEB, double ttfHighEB, double ttfLowEE, double ttfHighEE);

  static int nbMaxTowers_;
  static int nbMaxStrips_;
  static int nbMaxXtals_;
  static int nrMinTccEB_;
  static int nrMaxTccEB_;

 private:
  // updates Luts etc after change of parameters
  //FIXME: to be implemented for endcap also
  void update();

  int getIndex(int SM, int towerInSM, int stripInTower=0, int xtalInStrip=0) const ;

  double ttfLowEB_;
  double ttfHighEB_;
  double ttfLowEE_;
  double ttfHighEE_;

  double xtalLsbEB_ ;
  double xtalLsbEE_ ;
  double EtSatEB_ ;
  double EtSatEE_ ;

  std::map <int, std::vector<unsigned int> > towerParam_ ;
  std::map <int, std::vector<unsigned int> > stripParam_ ;
  std::map <int, std::vector<unsigned int> > xtalParam_ ;

 
};
#endif



