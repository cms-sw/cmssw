#ifndef EVENTFILTER_UTILITIES_ASCIIROLLINGCHART_H
#define EVENTFILTER_UTILITIES_ASCIIROLLINGCHART_H

#include "EventFilter/Utilities/interface/AsciiHisto.h"

#include <string>
#include <deque>

namespace evf{
  
  class AsciiRollingChart{
    
  public:
    AsciiRollingChart(std::string title, int nbins) : h_(title,nbins,0.,float(nbins)), roll_(nbins,0.)
    {
      
    }
    void flip(unsigned int ind, float x){
      if(ind==1){for(unsigned int i = 0; i < roll_.size(); i++) roll_[i]=0.;}
      if(ind<roll_.size())roll_[ind]=x;
      else{
	roll_.pop_front();
	roll_.push_back(x);
      }
      h_.fill(&(roll_.front()));
    }
    std::string &draw(){return h_.draw();}
      
  private:
    AsciiHisto h_;
    std::deque<float> roll_;
  };

}
#endif
