//
// $Id:$
//

#ifndef DataFormats_Centrality_h
#define DataFormats_Centrality_h


namespace reco { class Centrality {
public:
  Centrality(double energy = 0, int bin = -1, float npart = 0, float npart_sig = 0, float ncoll = 0, float ncoll_sig = 0, float b = 0, float b_sig = 0);
  virtual ~Centrality();

  double    HFEnergy()          const { return HFEnergy_; }
  int	    Bin()		const { return Bin_; }
  double    Npart()             const { return Npart_; } 
  double    Npart_sigma()       const { return Npart_sigma_; }
  double    Ncoll()             const { return Ncoll_; }
  double    Ncoll_sigma()       const { return Ncoll_sigma_; }
  double    imp_par()           const { return imp_par_; }
  double    imp_par_sigma()     const { return imp_par_sigma_; }  



private:
  double    HFEnergy_  ;
  int	    Bin_ ;       
  double    Npart_  ;
  double    Npart_sigma_ ;
  double    Ncoll_ ;
  double    Ncoll_sigma_ ;
  double    imp_par_ ;
  double    imp_par_sigma_ ;

};
}

#endif 


