#ifndef JetRejObs_h
#define JetRejObs_h

#include <vector>

class JetRejObs{
 public:
  JetRejObs(){};
  virtual ~JetRejObs(){};
  
  void setJetRejObs(std::vector<std::pair<int,double> > obs){ obs_ = obs; };
  unsigned int getSize() const { return obs_.size(); };
  std::pair<int, double> getPair(unsigned int i) const { return obs_[i]; };
  
 protected:
  std::vector<std::pair<int,double> > obs_;
};

#endif
