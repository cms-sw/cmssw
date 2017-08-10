///
/// Description: Simple Navigator class for the CaloTowers
///
///
///
/// \authors: Sam Harper - RAL, Olivier Davignon - Bristol
///

//
//

#ifndef L1Trigger_L1TCalorimeter_CaloStage2Nav_h
#define L1Trigger_L1TCalorimeter_CaloStage2Nav_h

#include "L1Trigger/L1TCalorimeter/interface/CaloTools.h"
#include <utility>
#include <cstdlib>

//allows us to move around in the Caloimeter from tower to tower
//invalid position is 0,0 no movement is possible for this
//0 is technically a valid phi position (0 for eta is really not) so we could have phi movement from such a position but I've decided that would be confusing have phi move and eta not.
namespace l1t{
  
  class CaloStage2Nav{
  public:

    CaloStage2Nav(); //defaults to 0,0 an invalid position
    CaloStage2Nav(int iEta,int iPhi);
    explicit CaloStage2Nav(std::pair<int,int> pos);
    
    
    //this function needs to ensure iPhi is range 1-72
    static int offsetIPhi(int iPhi,int offset){
      if(iPhi==0) return 0; //some debate here on whether I should accept 0 and just cast it to 72, I've decided that it is less confusing for 0,0 (the invalid position) to not move eta or phi rather than allowing phi to move
      else {
	iPhi+=offset;
	while(iPhi<=0) iPhi+=72;
	while(iPhi>72) iPhi-=72;
	return iPhi;
      }
    }

    //straight forward but have to watch the cases where we cross the 0 and/or 29 boundaries
    static int offsetIEta(int iEta,int offset)
{

      int etaMax = CaloTools::kHFEnd;
      int etaBoundaryHF = CaloTools::kHFBegin;

      if(iEta==0) return 0;//0 is not a valid position
      if(abs(iEta)>etaMax) return 0;//beyond +/-41 is not a valid position
      if(abs(iEta)==etaBoundaryHF) return 0;//+/-29 is not a valid position

      if(iEta>etaBoundaryHF)
	{
	  int iEta_tmp = iEta;

	  if(offset<0)
	    {
	      if(iEta_tmp+offset<=etaBoundaryHF) iEta_tmp--;
	      if(iEta_tmp+offset<=0) iEta_tmp--;
	      if(iEta_tmp+offset<=-etaBoundaryHF) iEta_tmp--;
	      if(iEta_tmp+offset<=-etaMax) return -etaMax;
	      return iEta_tmp+offset;
	    }
	  if(offset>=0)
	    {
	      if(iEta_tmp+offset>=etaMax) return etaMax;
	      return iEta_tmp+offset;
	    }
	}
      else if(iEta>0)
	{
	  int iEta_tmp = iEta;

	  if(offset<0)
	    {
	      if(iEta_tmp+offset<=0) iEta_tmp--;
	      if(iEta_tmp+offset<=-etaBoundaryHF) iEta_tmp--;
	      if(iEta_tmp+offset<=-etaMax) return -etaMax;
	      return iEta_tmp+offset;
	    }
	  if(offset>=0)
	    {
	      if(iEta_tmp+offset>=etaBoundaryHF) iEta_tmp++;
	      if(iEta_tmp+offset>=etaMax) return etaMax;
	      else return iEta_tmp+offset;
	    }
	}
      else if(iEta>-etaBoundaryHF)
	{
	  int iEta_tmp = iEta;

	  if(offset<0)
	    {
	      if(iEta_tmp+offset<=-etaBoundaryHF) iEta_tmp--;
	      if(iEta_tmp+offset<=-etaMax) return -etaMax;
	      return iEta_tmp+offset;
	    }
	  if(offset>=0)
	    {
	      if(iEta_tmp+offset>=0) iEta_tmp++;
	      if(iEta_tmp+offset>=etaBoundaryHF) iEta_tmp++;
	      if(iEta_tmp+offset>=etaMax) return etaMax;
	      return iEta_tmp+offset;
	    }
	}
      else
	{
	  int iEta_tmp = iEta;

	  if(offset<0)
	    {
	      if(iEta_tmp+offset<=-etaMax) return -etaMax;
	      return iEta_tmp+offset;
	    }
	  if(offset>=0)
	    {
	      if(iEta_tmp+offset>=-etaBoundaryHF) iEta_tmp++;
	      if(iEta_tmp+offset>=0) iEta_tmp++;
	      if(iEta_tmp+offset>=etaBoundaryHF) iEta_tmp++;
	      if(iEta_tmp+offset>=etaMax) return etaMax;
	      return iEta_tmp+offset;
	    }
	}
      return 0;
    }

    std::pair<int,int> offsetFromCurrPos(int iEtaOffset,int iPhiOffset)const;

    std::pair<int,int> move(int iEtaOffset,int iPhiOffset);
    std::pair<int,int> north(){return move(0,1);}
    std::pair<int,int> south(){return move(0,-1);}
    std::pair<int,int> east(){return move(1,0);}
    std::pair<int,int> west(){return move(-1,0);}

    std::pair<int,int> currPos()const{return currPos_;}
    int currIEta()const{return currPos().first;}
    int currIPhi()const{return currPos().second;}

    void resetPos(){currPos_=homePos_;}
    void resetIEta(){currPos_.first=homePos_.first;}
    void resetIPhi(){currPos_.second=homePos_.second;}
    void setHomePos(int iEta,int iPhi){homePos_.first=iEta;homePos_.second=iPhi;}
    void setHomePos(std::pair<int,int> pos){setHomePos(pos.first,pos.second);}

  private:
    std::pair<int,int> homePos_; //home position, can be reset to this
    std::pair<int,int> currPos_; //current position, null is 0,0 and no offseting or movement is possible from this (will return 0,0)
    

  };

}

#endif
