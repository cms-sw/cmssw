
//
// Levels:
//      
//    0   Neither recentering nor flattening is done.  The final weights are applied.
//    1   The sums over the sines and cosines are recentered.
//    2   Final results including both recentering and flattening.  Default if the level is not specified.
//    3   Calculation where all weights are set to unity.
//
//

#ifndef DataFormats_EvtPlane_h
#define DataFormats_EvtPlane_h

#include <vector>
#include <string>
#include <cmath>

namespace reco { class EvtPlane {
  public:
    EvtPlane(int epindx=0, int level = 2, double planeA=0,double sumSin=0, double sumCos=0, double sumw = 0, double sumw2 = 0, double pe = 0, double pe2 = 0, uint mult = 0);
    virtual ~EvtPlane();
    void addLevel(int level, double ang, double sumsin, double sumcos);
    int indx() const { return indx_ ;}
    float	angle(int level=2)   const { return (level>=0&&level<4)? angle_[level]:angle_[2]; }
    float	sumSin(int level=2)  const { return (level>=0&&level<4)? sumSin_[level]:sumSin_[2];}
    float	sumCos(int level=2)  const { return (level>=0&&level<4)? sumCos_[level]:sumCos_[2];}
    float	sumw() const { return sumw_;}
    float	sumw2() const { return sumw2_;}
    float	sumPtOrEt() const { return sumPtOrEt_;}
    float	sumPtOrEt2() const { return sumPtOrEt2_;}
    float	mult()    const { return mult_;}
    float	qy(int level=2) const { return sumSin(level); }
    float	qx(int level=2) const { return sumCos(level); }
    float	q(int level=2)  const { return ((pow(qx(level),2)+pow(qy(level),2))>0)? sqrt(pow(qx(level),2)+pow(qy(level),2)): 0.;}
    float	vn(int level=2) const{ return (q(level)>0 && fabs(sumw())>0)? q(level)/fabs(sumw()): 0.;}

  private:
    int		indx_;
    float	angle_[4];
    float	sumSin_[4];
    float	sumCos_[4];
    float	sumw_;
    float	sumw2_;
    float	sumPtOrEt_;
    float	sumPtOrEt2_;
    uint	mult_;

  };

  typedef std::vector<EvtPlane> EvtPlaneCollection;
}

#endif
