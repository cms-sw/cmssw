#ifndef __BAYES_STATS_H__
#define __BAYES_STATS_H__

class FCIntervals {
public:
  void Efficiency( const double pass, const double total,
		   double& mode, double& lowErr, double& highErr);
  
private:    
  
  double pass_;
  double total_;
  static double GetConfInterval() { return 0.683; }

  double SearchUpper (double low) const;
  double SearchLower (double low) const;
  double Brent (const double ax, const double bx, const double cx, const double tol, double &xmin) const;

  double Interval (double low) const;
  double Beta_ab (double a, double b, double k, double N) const;
  double Ibetai (double a, double b, double x) const;
};

#endif
