//-*-c++-*-
//-*-Weight.h-*-
//   Written by James Monk and Andrew Pilkington
/////////////////////////////////////////////////////////////////////////////
#ifndef WEIGHT_HH
#define WEIGHT_HH

#include <iostream>
#include <map>
#include <vector>

namespace Exhume{

  class Weight{

  public:

    Weight(){NPoints = 1000;};
    virtual ~Weight(){};
    inline std::map<double, double> GetFuncMap(){
      return(FuncMap);
    };
    inline double GetTotalIntegral(){
      return(TotalIntegral);
    };

    inline std::map<double, double> GetLineShape(){
      return(LineShape);
    };

  protected:

    virtual double WeightFunc(const double&)=0;

    void AddPoint(const double&, const double&);
    inline double GetFunc(const double &xx_){
      if(xx_ > Max_){
	return(WeightFunc(xx_) );
      }

      std::map<double, double>::iterator high_, low_;
      high_ = FuncMap.upper_bound(xx_);
      low_ = high_;
      low_--;

      return( low_->second + 
	      (high_->second - low_->second) * (xx_ - low_->first)/
	      (high_->first - low_->first));
    };

    inline double GetValue(const double &xx_){

      std::map<double, double>::iterator high_, low_;
      high_ = LineShape.upper_bound(xx_);
      
      if(high_==LineShape.end())high_--;

      low_ = high_;
      low_--;

      return( low_->second + 
	      (high_->second - low_->second) * (xx_ - low_->first)/
	      (high_->first - low_->first));
    };
    void WeightInit(const double&, const double&);

    double Max_;
    double TotalIntegral;

  private:

    unsigned int NPoints;
    std::map<double, double> FuncMap;
    std::map<double, double> LineShape;
  };
}



#endif
