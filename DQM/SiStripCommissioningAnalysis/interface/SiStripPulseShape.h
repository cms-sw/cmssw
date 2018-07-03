// Class:      SiStripPulseShape
//
/**\class SiStripPulseShape SiStripPulseShape.h myTestArea/SiStripPulseShape/src/SiStripPulseShape.h

 Description: analog pulse shape at the ouput of the APV. 

 Implementation:
     This class allows to access the pulse shape at the APV. This is  usefull to take into account
     timing effects in the tracker.
*/
//
// Original Author:  Christophe Delaere
//         Created:  Thu Nov  5 17:02:15 CEST 2006
// Revision Author:  Georg Auzinger
//         Created:  Thu Nov  5 17:02:15 CEST 2006
//         Updated:  Fri Jun  2 16:00:00 CEST 2017

//

#ifndef SiStripPulseShape_h_
#define SiStripPulseShape_h_

double fpeak(double *x, double *par);

double fdeconv(double *x, double *par);

double fpeak_convoluted(double *x, double *par);

double fdeconv_convoluted(double *x, double *par);

double pulse_raw(double x, double y, double z, double t);

double pulse_x0(double y, double z, double t);

double pulse_yz(double x, double z, double t);

double pulse_x0_yz(double z, double t);

double pulse(double x, double y, double z, double t);

double get_compensation(double x);


class SiStripPulseShape
{
  public: 
    enum mode {peak,deconvolution};
    SiStripPulseShape():mode_(deconvolution) {}
    virtual ~SiStripPulseShape() {}
    inline void setMode(const mode theMode) { mode_=theMode; }
    inline mode getMode() const { return mode_; } 
    inline double getNormalizedValue(const double& t) const
    {
      double parameters[5]={0.,-2.82,0.066,50,20};
      double time = t;
      switch(mode_) {
       case peak:
        {
	  return fpeak(&time,parameters);
	}
       case deconvolution:
        {
	  return fdeconv(&time,parameters);
	}
      }
    }
    
  private:
    mode mode_;
};

#endif
