#ifndef TopObjects_TtSemiMassSolution_h
#define TopObjects_TtSemiMassSolution_h
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TtSemiEvtSolution.h"
#include <vector>

class TtSemiMassSolution : public TtSemiEvtSolution
{
   public:
      TtSemiMassSolution();
      TtSemiMassSolution(TtSemiEvtSolution);
      virtual ~TtSemiMassSolution();
      
      void setMtopUncertainty(double);
      void setScanValues(std::vector<std::pair<double,double> >);
      
      double 	getMtopUncertainty() const	{ return dmtop; };
      std::vector<std::pair<double,double> > getScanValues() const 	{ return scanValues; };
        
   private:
      double dmtop;
      std::vector<std::pair<double,double> > scanValues;
};

#endif
