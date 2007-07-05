#ifndef SiPixelSCurveCalibration_SCurveContainer_h
#define SiPixelSCurveCalibration_SCurveContainer_h

/** \class SCurveContainer
 *
 * A class which contains the data and histograms
 * needed for an SCurve calibration.
 *
 * \authors Jason Keller (University of Nebraska)
 *
 * \version 1.2 July 5, 2007

 *
 ************************************************************/

#include <vector>

class SCurveContainer
{
  public:
    SCurveContainer();
    SCurveContainer(int vcalmin, int vcalmax, int vcalstep,
                      int ntriggers, int rowmax, int colmax,
                      int detid);
    ~SCurveContainer();

    double getEff(const int&, const int&, const int&) const;
    void setEff(const int&, const int&, const int&, const int&);
    unsigned int getRowMax() const {return rowmax_;}
    unsigned int getColMax() const {return colmax_;}
    unsigned int getRawId() const {return detid_;}

  private:
    unsigned int vcalmin_;
    unsigned int vcalmax_;
    unsigned int vcalstep_;
    unsigned int ntriggers_;
    unsigned int rowmax_;
    unsigned int colmax_;
    unsigned int detid_;

    std::vector<std::vector<std::vector<double> > > eff_;
};

#endif

