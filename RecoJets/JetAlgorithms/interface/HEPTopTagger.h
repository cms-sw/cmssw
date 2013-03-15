#ifndef __HEPTOPTAGGER_HH__
#define __HEPTOPTAGGER_HH__

class HEPTopTagger {
public:

  typedef fastjet::ClusterSequence ClusterSequence;
  typedef fastjet::JetAlgorithm JetAlgorithm;
  typedef fastjet::JetDefinition JetDefinition;
  typedef fastjet::PseudoJet PseudoJet;

  HEPTopTagger(const fastjet::ClusterSequence & cs,
	       const fastjet::PseudoJet & jet);

  HEPTopTagger(const fastjet::ClusterSequence & cs,
	       const fastjet::PseudoJet & jet,
	       double mtmass, double mwmass);

  void run_tagger();
  bool is_maybe_top() const {return _is_maybe_top;}
  bool is_masscut_passed() const {return _is_masscut_passed;}
  const PseudoJet & top_candidate() const {return _top_candidate;}
  const std::vector<PseudoJet> & top_subjets() const {return _top_subjets;}
  const std::vector<PseudoJet> & top_hadrons() const {return _top_hadrons;}
  unsigned top_count() const {return _top_count;}
  const std::vector<PseudoJet> & hardparts() const {return _top_parts;}
  unsigned parts_size() const {return _parts_size;}
  double delta_top() const {return _delta_top;}
  const std::vector<std::vector<PseudoJet> > & candjets() const {return _candjets;}
  void get_setting() const;
  void get_info() const;
  // for setting parameters
  void set_max_subjet_mass(double x) {_max_subjet_mass=x;}
  void set_mass_drop_threshold(double x) {_mass_drop_threshold=x;}
  void set_top_range(double xmin, double xmax) {_mtmin=xmin; _mtmax=xmax;}
  void set_mass_ratio_range(double rmin, double rmax) {_rmin=rmin; _rmax=rmax;}
  void set_mass_ratio_cut(double m23cut, double m13cutmin,double m13cutmax){_m23cut=m23cut; _m13cutmin=m13cutmin; _m13cutmax=m13cutmax;}
  void set_nfilt(unsigned nfilt) {_nfilt=nfilt;}
  void set_filtering_jetalgorithm(JetAlgorithm jet_algorithm) {_jet_algorithm=jet_algorithm;}
  void set_reclustering_jetalgorithm(JetAlgorithm jet_algorithm) {_jet_algorithm_recluster=jet_algorithm;}
  // 
  double cos_theta_h() const;
  double dr_bjj() const;
  std::vector<double> dr_values() const;

private:
  const ClusterSequence * _cs;
  const PseudoJet _jet;
  const double _mtmass, _mwmass;
  double _mass_drop_threshold;
  double _max_subjet_mass; // stop when subjet mass < 30 GeV
  double _mtmin, _mtmax;
  double _rmin, _rmax;
  double _m23cut, _m13cutmin, _m13cutmax;
  size_t _nfilt;
  // filtering algorithm
  JetAlgorithm _jet_algorithm;
  JetAlgorithm _jet_algorithm_recluster;
  
  bool _is_masscut_passed;
  bool _is_maybe_top;
  double _delta_top;
  unsigned _top_count;
  unsigned _parts_size;
  PseudoJet _top_candidate;
  std::vector<PseudoJet> _top_subjets;
  std::vector<PseudoJet> _top_hadrons;
  std::vector<PseudoJet> _top_parts;
  std::vector<std::vector<PseudoJet> > _candjets;

  void FindHardSubst(const PseudoJet& jet, std::vector<fastjet::PseudoJet>& t_parts);
  std::vector<PseudoJet> Filtering(const std::vector <PseudoJet> & top_constits, const JetDefinition & filtering_def);
  void store_topsubjets(const std::vector<PseudoJet>& top_subs);
  bool check_mass_criteria(const std::vector<fastjet::PseudoJet> & top_subs) const;
  double check_cos_theta(const PseudoJet & jet, const PseudoJet & subj1,const PseudoJet & subj2) const;
  PseudoJet Sum(const std::vector<PseudoJet>& );
  double r_max_3jets(const fastjet::PseudoJet & jet1,const fastjet::PseudoJet & jet2,
		     const fastjet::PseudoJet & jet3) const;

  bool debugg;

};
//--------------------------------------------------------------------
double HEPTopTagger::cos_theta_h() const {
  return check_cos_theta(_top_candidate,_top_subjets[1],_top_subjets[2]);// m23 is closest to mW
}

double HEPTopTagger::dr_bjj() const{
  if(_top_subjets.size()!=3){return -1;}
  return r_max_3jets(_top_subjets[0],_top_subjets[1],_top_subjets[2]);
}

std::vector<double> HEPTopTagger::dr_values() const{
  std::vector<double> dr_values;
  dr_values.push_back(sqrt(_top_subjets[1].squared_distance(_top_subjets[2])));
  dr_values.push_back(sqrt(_top_subjets[0].squared_distance(_top_subjets[2])));
  dr_values.push_back(sqrt(_top_subjets[0].squared_distance(_top_subjets[1])));
  return dr_values;
}


double HEPTopTagger::r_max_3jets(const fastjet::PseudoJet & jet1,const fastjet::PseudoJet & jet2,const fastjet::PseudoJet & jet3) const{
  fastjet::PseudoJet jet12,jet13,jet23;
  jet12=jet1+jet2;
  jet13=jet1+jet3;
  jet23=jet2+jet3;

  double a=sqrt(jet1.squared_distance(jet2));
  double b=sqrt(jet2.squared_distance(jet3));
  double c=sqrt(jet3.squared_distance(jet1));
  double dR1=a,dR2=a;

  if(a<=b && a<=c){
    dR1=a;
    dR2=sqrt(jet12.squared_distance(jet3));
  };
  if(b<a && b<=c){
    dR1=b;
    dR2=sqrt(jet23.squared_distance(jet1));
  };
  if(c<a && c<b){
    dR1=c;
    dR2=sqrt(jet13.squared_distance(jet2));
  };
  return max(dR1,dR2);
}

double HEPTopTagger::check_cos_theta(const PseudoJet & jet,const PseudoJet & subj1,const PseudoJet & subj2) const
{
  // the two jets of interest: top and lower-pt prong of W
  PseudoJet W2;
  PseudoJet top = jet;
  
  if(subj1.perp2() < subj2.perp2())
    {
      W2 = subj1;
    }
  else
    {
      W2 = subj2;
    }

  // transform these jets into jets in the rest frame of the W
  W2.unboost(subj1+subj2);
  top.unboost(subj1+subj2);
  
  double csthet = (W2.px()*top.px() + W2.py()*top.py() + W2.pz()*top.pz())/sqrt(W2.modp2() * top.modp2());  
  return(csthet);
}

void HEPTopTagger::FindHardSubst(const PseudoJet & this_jet, std::vector<fastjet::PseudoJet> & t_parts)
{
  PseudoJet parent1(0,0,0,0), parent2(0,0,0,0);
  if (this_jet.m() < _max_subjet_mass || !_cs->has_parents(this_jet, parent1, parent2))
    {
      t_parts.push_back(this_jet);
    }
  else 
    {
      if (parent1.m() < parent2.m()) swap(parent1, parent2);
      
      FindHardSubst(parent1,t_parts);
      
      if (parent1.m() < _mass_drop_threshold * this_jet.m())
	{
	  FindHardSubst(parent2,t_parts);
	}
    }
}

void HEPTopTagger::store_topsubjets(const std::vector<PseudoJet>& top_subs){
  _top_subjets.resize(0);
  double m12=(top_subs[0]+top_subs[1]).m();
  double m13=(top_subs[0]+top_subs[2]).m();
  double m23=(top_subs[1]+top_subs[2]).m();
  //double m123=(top_subs[0]+top_subs[1]+top_subs[2]).m();
  double dm12=abs(m12-_mwmass);
  double dm13=abs(m13-_mwmass);
  double dm23=abs(m23-_mwmass);
  //double dm_min=min(dm12,min(dm13,dm23));
  if(dm23<=dm12 && dm23<=dm13){
    _top_subjets.push_back(top_subs[0]); //supposed to be b
    _top_subjets.push_back(top_subs[1]); //W-jet 1
    _top_subjets.push_back(top_subs[2]); //W-jet 2		
  }
  else if(dm13<=dm12 && dm13<dm23){
    _top_subjets.push_back(top_subs[1]); //supposed to be b
    _top_subjets.push_back(top_subs[0]); //W-jet 1
    _top_subjets.push_back(top_subs[2]); //W-jet 2
  }
  else if(dm12<dm23 && dm12<dm13){
    _top_subjets.push_back(top_subs[2]); //supposed to be b
    _top_subjets.push_back(top_subs[0]); //W-jet 1
    _top_subjets.push_back(top_subs[1]); //W-jet 2
  }
  return;
}

bool HEPTopTagger::check_mass_criteria(const std::vector<PseudoJet> & top_subs) const{
  bool is_passed=false;
  double m12=(top_subs[0]+top_subs[1]).m();
  double m13=(top_subs[0]+top_subs[2]).m();
  double m23=(top_subs[1]+top_subs[2]).m();
  double m123=(top_subs[0]+top_subs[1]+top_subs[2]).m();
  if(
     (atan(m13/m12)>_m13cutmin && _m13cutmax > atan(m13/m12)
      && (m23/m123>_rmin && _rmax>m23/m123))
     ||
     (((m23/m123)*(m23/m123) < 1-_rmin*_rmin*(1+(m13/m12)*(m13/m12))) &&
      ((m23/m123)*(m23/m123) > 1-_rmax*_rmax*(1+(m13/m12)*(m13/m12))) && 
      (m23/m123 > _m23cut))
     ||
     (((m23/m123)*(m23/m123) < 1-_rmin*_rmin*(1+(m12/m13)*(m12/m13))) &&
      ((m23/m123)*(m23/m123) > 1-_rmax*_rmax*(1+(m12/m13)*(m12/m13))) && 
      (m23/m123 > _m23cut))
     ){ 
    is_passed=true;
  }
  return is_passed;
}

////////// Top-TAGGER: /////////////////////////////////////////////////////////////////
HEPTopTagger::HEPTopTagger(const fastjet::ClusterSequence & cs,
			   const fastjet::PseudoJet & jet) : 
  _cs(&cs), _jet(jet), _mtmass(172.3), _mwmass(80.4), 
  _mass_drop_threshold(0.8), _max_subjet_mass(30.),
  _mtmin(172.3 - 25.),_mtmax(172.3 + 25.), _rmin(0.85*80.4/172.3),_rmax(1.15*80.4/172.3),
  _m23cut(0.35),_m13cutmin(0.2),_m13cutmax(1.3),
  _nfilt(5),_jet_algorithm(fastjet::cambridge_algorithm),_jet_algorithm_recluster(fastjet::cambridge_algorithm),
  debugg(false)
{}

HEPTopTagger::HEPTopTagger(const fastjet::ClusterSequence & cs,
			   const fastjet::PseudoJet & jet,
			   double mtmass,double mwmass
			   ) : 
  _cs(&cs), _jet(jet), _mtmass(mtmass), _mwmass(mwmass), 
  _mass_drop_threshold(0.8), _max_subjet_mass(30.),
  _mtmin(mtmass - 25.),_mtmax(mtmass + 25.), _rmin(0.85*mwmass/mtmass),_rmax(1.15*mwmass/mtmass),
  _m23cut(0.35),_m13cutmin(0.2),_m13cutmax(1.3),
  _nfilt(5),_jet_algorithm(fastjet::cambridge_algorithm),_jet_algorithm_recluster(fastjet::cambridge_algorithm),
  debugg(false)
{}


void HEPTopTagger::run_tagger()
{
  _delta_top=1000000000000.0;
  _top_candidate.reset(0.,0.,0.,0.);
  _top_count=0;
  _parts_size=0;
  _is_maybe_top=_is_masscut_passed=false;
  _top_subjets.clear();
  _top_hadrons.clear();
  _top_parts.clear();

  if(debugg)
    {
      cout << "mtmass in top_tagger: " << _mtmass << endl;
      cout << "mwmass in top_tagger: " << _mwmass << endl;
      cout << "jet input HEPTopTagger: " << endl;
      //printjet(_jet);
    }
  
  
  // input this_jet, output _top_parts
  FindHardSubst(_jet, _top_parts);
  
  // store hard substructure of the top candidate
  _parts_size=_top_parts.size();
  
  // these events are not interesting 
  if(_top_parts.size() < 3){return;}
  
  for(unsigned rr=0; rr<_top_parts.size(); rr++){
    for(unsigned ll=rr+1; ll<_top_parts.size(); ll++){
      for(unsigned kk=ll+1; kk<_top_parts.size(); kk++){
	// define top_constituents candidate before filtering 	      
	std::vector <PseudoJet> top_constits = _cs->constituents(_top_parts[rr]);
	_cs->add_constituents(_top_parts[ll],top_constits);
	_cs->add_constituents(_top_parts[kk],top_constits);	      

	      // define Filtering: filt_top_R and jetdefinition 
	double filt_top_R 
	  = min(0.3,0.5*sqrt(min(_top_parts[kk].squared_distance(_top_parts[ll]),
				 min(_top_parts[rr].squared_distance(_top_parts[ll]),
				     _top_parts[kk].squared_distance(_top_parts[rr])))));
	JetDefinition filtering_def(_jet_algorithm, filt_top_R);
	std::vector<PseudoJet> top_constits_filtered = Filtering(top_constits,filtering_def);
	PseudoJet topcandidate = Sum(top_constits_filtered);
	if( topcandidate.m() < _mtmin || _mtmax < topcandidate.m() ) continue;
	_top_count++;
	// obtain 3 subjets
	JetDefinition reclustering(_jet_algorithm_recluster, 3.14/2);
	
     //// **** NEXT 3 LINES EDITED CKV 12/2/12 **** (edit suggested by G. P. Salam)
     ClusterSequence * cssubtop = new ClusterSequence(top_constits_filtered,reclustering);
	std::vector <PseudoJet> top_subs = sorted_by_pt(cssubtop->exclusive_jets(3));	      
	cssubtop->delete_self_when_unused();
     //// **** END EDIT ***************************
     
     _candjets.push_back(top_subs); //
	
	// transfer infos of the positively identified top to the outer world 
	double deltatop = abs(topcandidate.m() - _mtmass);
	if(deltatop < _delta_top){	 
	  _delta_top = deltatop;
	  _is_maybe_top = true;
	  _top_candidate = topcandidate;
	  store_topsubjets(top_subs);
	  _top_hadrons=top_constits_filtered;
	  /////////////////////// check mass plane cut////////////////////////
	  _is_masscut_passed=check_mass_criteria(top_subs);
	}// end deltatop < _delta_top
      }// end kk
    }// end ll
  }// end rr
  return;
}


std::vector<fastjet::PseudoJet> HEPTopTagger::Filtering(const std::vector <PseudoJet> & top_constits, const JetDefinition & filtering_def)
{
  // perform filtering
  fastjet::ClusterSequence cstopfilt( top_constits, filtering_def);
 
  // extract top subjets
  std::vector<PseudoJet> filt_top_subjets = sorted_by_pt(cstopfilt.inclusive_jets());
  
  // take first n_topfilt subjets
  std::vector<PseudoJet> top_constits_filtered;
  for(unsigned ii = 0; ii<min(_nfilt, filt_top_subjets.size()) ; ii++)
    {
      cstopfilt.add_constituents(filt_top_subjets[ii],top_constits_filtered);
    }
  return top_constits_filtered;
}


fastjet::PseudoJet HEPTopTagger::Sum(const std::vector<PseudoJet> & vec_pjet)
{
  PseudoJet sum;
  sum.reset(0.,0.,0.,0.);
  for(unsigned i=0;i<vec_pjet.size();i++){
    sum += vec_pjet.at(i);
  }
  return sum;
}

void HEPTopTagger::get_info() const
{
  cout << "maybe_top: " <<  _is_maybe_top << endl;
  cout << "mascut_passed: " <<  _is_masscut_passed << endl;
  cout << "top candidate mass:" <<  _top_candidate.m() << endl;
  cout << "top candidate (pt, eta, phi): (" 
       <<  _top_candidate.perp() << ","
       <<  _top_candidate.eta() << ","
       <<  _top_candidate.phi_std() << ")" << endl;
  cout << "hadrons size: " <<  _top_hadrons.size() << endl;
  cout << "topcount: " <<  _top_count << endl;
  cout << "parts size: " <<  _parts_size << endl;
  cout << "delta_top: " <<  _delta_top << endl;  
  return;
}


void HEPTopTagger::get_setting() const
{
  cout << "top mass: " <<  _mtmass << endl;
  cout << "W mass: " <<  _mwmass << endl;
  cout << "top mass range: [" << _mtmin << ", " << _mtmax << "]" << endl;
  cout << "W mass ratio range: [" << _rmin << ", " << _rmax << "] (["
       <<_rmin*_mtmass/_mwmass<< "%, "<< _rmax*_mtmass/_mwmass << "%])"<< endl;
  cout << "mass ratio cut: (m23cut, m13min, m13max)=(" 
       << _m23cut << ", " << _m13cutmin << ", " << _m13cutmax << ")" << endl;
  cout << "mass_drop_threshold: " << _mass_drop_threshold << endl;
  cout << "max_subjet_mass: " << _max_subjet_mass << endl;
  cout << "n_filtering: " << _nfilt << endl;
  cout << "JetAlgorithm for filtering: "<< _jet_algorithm << endl;
  cout << "JetAlgorithm for reclustering: "<< _jet_algorithm_recluster << endl;
  return;
}


#endif // __HEPTOPTAGGER_HH__
