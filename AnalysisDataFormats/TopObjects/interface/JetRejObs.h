#ifndef JetRejObs_h
#define JetRejObs_h

#include <vector>


using namespace std;

class JetRejObs{
   
   public:
      JetRejObs();
      virtual ~JetRejObs();
            
      void 		setJetRejObs(vector<pair<int,double> >);
      
      unsigned int 	        getSize() const;
      pair<int, double>  	getPair(unsigned int) const;
      
   protected:
      vector<pair<int,double> >  obs;
    
      
};


#endif
