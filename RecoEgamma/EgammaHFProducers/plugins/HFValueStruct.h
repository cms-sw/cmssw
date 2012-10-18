#ifndef _HFVALUESTRUCT_H
#define _HFVALUESTRUCT_H

#include <iostream>
#include <vector>
#include <Rtypes.h>


namespace reco {
  
  
  class HFValueStruct {
  public:
    

    
    HFValueStruct() {}
    HFValueStruct(const int& version, const std::vector<double>& vect);
    // returns single value by index
    
      double EnCor(int ieta)const;
      double PUSlope(int ieta)const;
      double PUIntercept(int ieta)const;
      
// sets single value by index
      void setEnCor(int ieta,double val);
      void setPUSlope(int ieta,double val);
      void setPUIntercept(int ieta,double val);
   

    
    // returns whole vector
      std::vector<double> EnCor()const;
      std::vector<double> PUSlope()const;
      std::vector<double> PUIntercept()const;
      
      // set whole vector
      void setEnCor(const std::vector<double>& val);
      void setPUSlope(const std::vector<double>& val);
      void setPUIntercept(const std::vector<double>& val);
      
      
      
  private:
      int v_;
      std::vector<double> hfvv_;
      //std::vector<double> SetHfvvFromDB_();  //will need when in database
      bool doEnCor_,doPU_;
      
      int indexByIeta(int& ieta)const;
      int ietaByIndex(int& indx)const;
      
  };
}
#endif
