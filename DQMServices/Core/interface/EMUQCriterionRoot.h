#include "DQMServices/Core/interface/QTestStatus.h"
#include "DQMServices/Core/interface/QCriterion.h"
#include "DQMServices/Core/interface/MonitorElementT.h"

#include "DQMServices/Core/interface/RuleAllContentWithinFixedRange.h"
#include "DQMServices/Core/interface/RuleAllContentWithinFloatingRange.h"
#include "DQMServices/Core/interface/RuleAllContentAlongDiagonal.h"
#include "DQMServices/Core/interface/RuleFlatOccupancy1d.h"
#include "DQMServices/Core/interface/RuleFixedFlatOccupancy1d.h"
#include "DQMServices/Core/interface/RuleCSC01.h"

#include <sstream>
#include <string>

#ifndef _EMUQCRITERION_ROOT_H
#define _EMUQCRITERION_ROOT_H

class  MEAllContentWithinFixedRangeROOT : public QCriterionRoot<TH1F>, public RuleAllContentWithinFixedRange
{
 public:
  MEAllContentWithinFixedRangeROOT(std::string name) : QCriterionRoot<TH1F>(name), RuleAllContentWithinFixedRange() 
  {
    setAlgoName(RuleAllContentWithinFixedRange::getAlgoName());
  }
  ~MEAllContentWithinFixedRangeROOT(void){}
  
  std::string getAlgoName(void)
  {return RuleAllContentWithinFixedRange::getAlgoName();}

  float runTest(const TH1F * const t)
    { 
      //double x, y, z; 
      set_x_min( 6.0 );
      set_x_max( 9.0 );
      set_epsilon_max( 0.1 );
      set_S_fail( 5.0 );
      set_S_pass( 5.0 );
      return RuleAllContentWithinFixedRange::runTest(t);//, 1.0, 1.0, 0.1, 5.0, 5.0, &x, &y, &z);
      //cout << "My test: " << get_epsilon_obs << " " << get_S_fail_obs() << " " << get_S_pass_obs << endl;
      //return result;
    }

  const TH1F * getObject(const MonitorElement * const me) const
    {return QCriterionRoot<TH1F>::getROOTObject(me);}

  bool notEnoughStats(const TH1F * const t) const
  {return RuleAllContentWithinFixedRange::notEnoughStats(t);}

 protected:
  bool isInvalid(const MonitorElement * const me)
  {return false;} // any scenarios for invalid test?
  void setMessage(void)
  {
    std::ostringstream message;
    message << " Test " << qtname_ << " (" << getAlgoName() 
	    << "): Result = " << prob_;
    message_ = message.str();      
  }
};

class  MEAllContentWithinFloatingRangeROOT : public QCriterionRoot<TH1F>, public RuleAllContentWithinFloatingRange
{
 public:
  MEAllContentWithinFloatingRangeROOT(std::string name) : QCriterionRoot<TH1F>(name), RuleAllContentWithinFloatingRange() 
  {
    setAlgoName(RuleAllContentWithinFloatingRange::getAlgoName());
  }
  ~MEAllContentWithinFloatingRangeROOT(void){}
  
  std::string getAlgoName(void)
  {return RuleAllContentWithinFloatingRange::getAlgoName();}

  float runTest(const TH1F * const t)
    { 
      //double x, y, z; 
      set_Nrange( 1 );
      set_epsilon_max( 0.1 );
      set_S_fail( 5.0 );
      set_S_pass( 5.0 );
      return RuleAllContentWithinFloatingRange::runTest(t);
      //cout << "My test: " << get_epsilon_obs << " " << get_S_fail_obs() << " " << get_S_pass_obs << endl;
      //return result;
    }

  const TH1F * getObject(const MonitorElement * const me) const
    {return QCriterionRoot<TH1F>::getROOTObject(me);}

  bool notEnoughStats(const TH1F * const t) const
  {return RuleAllContentWithinFloatingRange::notEnoughStats(t);}

 protected:
  bool isInvalid(const MonitorElement * const me)
  {return false;} // any scenarios for invalid test?
  void setMessage(void)
  {
    std::ostringstream message;
    message << " Test " << qtname_ << " (" << getAlgoName() 
	    << "): Result = " << prob_;
    message_ = message.str();      
  }
};

class  MEFixedFlatOccupancy1dROOT : public QCriterionRoot<TH1F>, public RuleFixedFlatOccupancy1d
{
 public:
  MEFixedFlatOccupancy1dROOT(std::string name) : QCriterionRoot<TH1F>(name), RuleFixedFlatOccupancy1d() 
  {
    setAlgoName(RuleFixedFlatOccupancy1d::getAlgoName());
  }
  ~MEFixedFlatOccupancy1dROOT(void){}
  
  std::string getAlgoName(void)
  {return RuleFixedFlatOccupancy1d::getAlgoName();}

  float runTest(const TH1F * const t)
    { 
      set_Occupancy( 1.0 );
      double mask[10] = {1,0,0,0,1,1,1,1,1,1};
      set_ExclusionMask( mask );
      set_epsilon_min( 0.099 );
      set_epsilon_max( 0.101 );
      set_S_fail( 5.0 );
      set_S_pass( 5.0 ); 
      return RuleFixedFlatOccupancy1d::runTest(t);
      //cout << "My test: " << get_epsilon_obs << " " << get_S_fail_obs() << " " << get_S_pass_obs << endl;
      //return result;
    }

  const TH1F * getObject(const MonitorElement * const me) const
    {return QCriterionRoot<TH1F>::getROOTObject(me);}

  bool notEnoughStats(const TH1F * const t) const
  {return RuleFixedFlatOccupancy1d::notEnoughStats(t);}

 protected:
  bool isInvalid(const MonitorElement * const me)
  {return false;} // any scenarios for invalid test?
  void setMessage(void)
  {
    std::ostringstream message;
    message << " Test " << qtname_ << " (" << getAlgoName() 
	    << "): Result = " << prob_;
    message_ = message.str();      
  }
};

class  MECSC01ROOT : public QCriterionRoot<TH1F>, public RuleCSC01
{
 public:
  MECSC01ROOT(std::string name) : QCriterionRoot<TH1F>(name), RuleCSC01() 
  {
    setAlgoName(RuleCSC01::getAlgoName());
  }
  ~MECSC01ROOT(void){}
  
  std::string getAlgoName(void)
  {return RuleCSC01::getAlgoName();}

  float runTest(const TH1F * const t)
    { 
      set_epsilon_max( 0.1 );
      set_S_fail( 5.0 );
      set_S_pass( 5.0 );
      return RuleCSC01::runTest(t);
      //cout << "My test: " << get_epsilon_obs << " " << get_S_fail_obs() << " " << get_S_pass_obs << endl;
      //return result;
    }

  const TH1F * getObject(const MonitorElement * const me) const
    {return QCriterionRoot<TH1F>::getROOTObject(me);}

  bool notEnoughStats(const TH1F * const t) const
  {return RuleCSC01::notEnoughStats(t);}

 protected:
  bool isInvalid(const MonitorElement * const me)
  {return false;} // any scenarios for invalid test?
  void setMessage(void)
  {
    std::ostringstream message;
    message << " Test " << qtname_ << " (" << getAlgoName() 
	    << "): Result = " << prob_;
    message_ = message.str();      
  }
};

#endif
