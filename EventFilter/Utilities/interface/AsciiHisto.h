#ifndef EVENTFILTER_UTILITIES_ASCIIHISTO_H
#define EVENTFILTER_UTILITIES_ASCIIHISTO_H

#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <vector>

namespace evf{
  
  class AsciiHisto{
    
  public:
    AsciiHisto(std::string title, unsigned int nbins,float min, float max) : 
      uptodate_(false),  
      nbins_(nbins), xmin_(min), xmax_(max), ymax_(-100000.), cont_(nbins_,0), title_(title)
    {
      float span = xmax_-xmin_;
      binsize_ = span/nbins_;
    }
    
    void fill(float x)
    {
      uptodate_ = false;
      int bin = (x-xmin_)/binsize_;
      cont_[bin]++;
    }
    void fill(float *cont)
    {
      uptodate_ = false;
      for(unsigned int i = 0; i < cont_.size(); i++)
	cont_[i] = cont[i];
    }
    
    int maxbin()
    {
      int retval = -1;
      ymax_ = -100000.;
      for(unsigned int i = 0; i < nbins_; i++)
	if(ymax_ < cont_[i]){ymax_ = cont_[i]; retval = i;}
/*       std::cout << "max bin " << retval << " has " << ymax_ << std::endl; */
/*       std::cout << "nbins " << nbins_  */
/* 		<< " binsize " << binsize_ << std::endl; */
      return retval;
    }
    std::string &draw()
    {
      if(uptodate_) return current_;
      maxbin();
      float yscale = (ymax_*1.2) /20.;
      std::ostringstream ost;
      ost << "         ^"; 
      ost << std::setw(nbins_-title_.size()) << " ";                
      ost << title_ << std::endl;
      //      std::cout << "         ^" <<  "                " << title_ << std::endl;
      for(int j = 20; j>=0; j--)
	{
	  //	  std::cout << "--------------> "<< j << std::endl;
	  if(j%5==0){
	    ost << std::setw(8) << yscale*j << "-|";
	    //	    std::cout << std::setw(8) << yscale*j << "-|";
	  }
	  else{
	    ost << "         |";
	    //	    std::cout  << "         |";
	  }
	  for(unsigned int i = 0; i < nbins_+5; i++)
	    {
	      if(j==0) {ost << "-"; /*std::cout << "-";*/}
	      else
		{
		  if(i<nbins_ && cont_[i] > yscale*j) 
		    {ost << "*"; /*std::cout << "*";*/}
		  else
		    {ost << " "; /*std::cout << " ";*/}
		}
	    }
	  if(j==0){ost << ">"; /*std::cout << ">";*/}
	  ost << std::endl;
	  //	  std::cout << std::endl;
	}
      ost << "         ";
      //      std::cout  << "         ";

      for(unsigned int i = 0; i < nbins_+5; i++)
	{
	  if(i%10==0)
	    {ost << "|"; /*std::cout << "|";*/}
	  else
	    {ost << " "; /*std::cout << " ";*/}
	}
      ost << std::endl;
      //      std::cout << std::endl;
      ost << std::setw(10) << xmin_ ;
      //      std::cout << std::setw(10) << xmin_; 
      for(unsigned int i = 0; i < nbins_+5; i++)
	if((i+3)%10==0){ost << std::setw(10) << xmin_+binsize_*(i+3); 
	  //	  std::cout << std::setw(10) << xmin_+binsize_*(i+3); 
	}
      ost << std::endl;
      //      std::cout << std::endl;
      
      current_ = ost.str();
      return current_;
    }

  private:
    bool uptodate_;
    unsigned int nbins_;
    float binsize_;
    float xmin_;
    float xmax_;
    float ymax_;
    float ymin_;
    std::vector<float> cont_;
    std::string current_;
    std::string title_;
  };

}
#endif
