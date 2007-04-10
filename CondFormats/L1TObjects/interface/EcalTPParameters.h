#ifndef CondFormats_L1TObjects_EcalTPParameters_H
#define CondFormats_L1TObjects_EcalTPParameters_H
/**
 * Author: P.Paganini, Ursula Berthon
 * Created: 20 March 2007
 * $Id: EcalTPParameters.h,v 1.1 2007/03/01 18:18:24 uberthon Exp $
 **/

#include <vector>
#include <map>


class EcalTPParameters {
 public:
   
  EcalTPParameters();
  ~EcalTPParameters();

  // getters
  std::vector<unsigned int> getTowerParameters(int SM, int towerInSM, bool print = false) const ;
  std::vector<unsigned int> getStripParameters(int SM, int towerInSM, int stripInTower, bool print = false) const ;
  std::vector<unsigned int> getXtalParameters(int SM, int towerInSM, int stripInTower, int xtalInStrip, bool print = false)  const;

  double getTPGLsbEB(int compressedEt) const {return 0.469;}  // GeV   //FIXME: to be implemented
  double getTPGLsbEE(int compressedEt) const {return 0.560;} // GeV  //FIXME: to be implemented
  double getEtSatEB() const {return EtSatEB_;}
  double getEtSatEE() const {return EtSatEE_;}

  // setters
  void setTowerParameters(int SM, int towerInSM, std::vector<unsigned int> params) 	    
    {towerParam_[getIndex(SM,towerInSM)]=params;}         
  void setStripParameters(int SM, int towerInSM, int stripInTower, std::vector<unsigned int> params)  
    {stripParam_[getIndex(SM,towerInSM,stripInTower)]=params;}
  void setXtalParameters(int SM, int towerInSM, int stripInTower, int xtalInStrip, std::vector<unsigned int> params) 
    {xtalParam_[getIndex(SM,towerInSM,stripInTower,xtalInStrip)]=params;}		

  //  void setTowerParameters(int index, std::vector<unsigned int> params) { towerParam_[index]=params;} 
  //  void setStripParameters(int index, std::vector<unsigned int> params) { stripParam_[index]=params;} 
  //  void setXtalParameters(int index, std::vector<unsigned int> params) { xtalParam_[index]=params;} 

  void changeThresholds(double ttfLowEB, double ttfHighEB, double ttfLowEE, double ttfHighEE);

 private:
  // updates Luts etc after change of parameters
  //FIXME: to be implemented, at least for changing according to  TTF thresholds
  void update() {;}

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



