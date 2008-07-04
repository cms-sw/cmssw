#ifndef PixelDACScanRange_h
#define PixelDACScanRange_h
//
// This class collects the information
// about the range of DAC settings used
// in scans of the DACs.
//
//
//

#include <string>

namespace pos{
  class PixelDACScanRange {

  public:

    PixelDACScanRange(){;}
    PixelDACScanRange(std::string dacname, unsigned int first, 
		      unsigned int last, unsigned int step,
		      unsigned int index);

    std::string name() const { return name_;}
    unsigned int dacchannel() const { return dacchannel_; }
    unsigned int step() const { return step_; }
    unsigned int first() const { return first_; }
    unsigned int last() const { return last_; }
    unsigned int index() const { return index_; }
    unsigned int getNPoints() const { return (last_-first_)/step_+1; }

  private:

    std::string name_;
    unsigned int dacchannel_;
    unsigned int first_;
    unsigned int last_;
    unsigned int step_;
    unsigned int index_;


  };
}
#endif
