#include "RecoJets/JetAlgorithms/interface/Qjets.h"

Qjets::Qjets(double zcut, double dcut_fctr, double exp_min, double exp_max, double rigidity)
  : _zcut(zcut), _dcut_fctr(dcut_fctr), _exp_min(exp_min), _exp_max(exp_max), _rigidity(rigidity), _dcut(-1.), _rand_seed_set(false)
{
}

void Qjets::SetRandSeed(unsigned int seed){
  _rand_seed_set = true;
  _seed = seed;
}

bool Qjets::JetUnmerged(int num){
  if(count(_merged_jets.begin(), _merged_jets.end(), num) > 0)
    return false;
  return true;
}

bool Qjets::JetsUnmerged(jet_distance& jd){
  return JetUnmerged(jd.j1) && JetUnmerged(jd.j2);
}

void Qjets::ComputeNewDistanceMeasures(fastjet::ClusterSequence & cs, int new_jet){
  // jet-jet distances
  for(unsigned int i = 0; i < cs.jets().size(); i++)
    if(JetUnmerged(i) && i != (unsigned int) new_jet){
      jet_distance jd;
      jd.j1 = new_jet;
      jd.j2 = i;
      jd.dij = d_ij(cs.jets()[jd.j1], cs.jets()[jd.j2]);
      _distances.push_back(jd);
    }
}

void Qjets::ComputeDCut(fastjet::ClusterSequence & cs){
  // assume all jets in cs form a single jet.  compute mass and pt
  fastjet::PseudoJet sum(0.,0.,0.,0.);
  for(vector<fastjet::PseudoJet>::const_iterator it = cs.jets().begin(); it != cs.jets().end(); it++)
    sum += (*it);
  _dcut = 2. * _dcut_fctr * sum.m()/sum.perp(); 
}

double Qjets::ComputeMinimumDistance(){
  double dmin(-1.);
  for(list<jet_distance>::iterator it = _distances.begin(); it != _distances.end(); )
    if(JetsUnmerged(*it)){
      if(dmin == -1. || (*it).dij < dmin)
	dmin = (*it).dij;	      
      it++;
    } else
      it = _distances.erase(it);    
  return dmin;
}

double Qjets::ComputeNormalization(double dmin){
  double norm(0.);
  for(list<jet_distance>::iterator it = _distances.begin(); it != _distances.end(); )
    if(JetsUnmerged(*it)){
      double inc = 0.;
      /*
	We need to be careful about dmin == 0 which happens for collinear jets.
	(Thanks go to Nhan V Tran for finding this bug)
      */
      if(dmin == 0.){
	if((*it).dij == 0.)
	  inc = 1.;
	else
	  inc = 0.;
      } else 
	inc = exp(-_rigidity*((*it).dij-dmin)/dmin);      
      
      assert(inc <= 1. && !isnan((float) inc));      
      norm += inc;          
      it++;
    } else 
      it = _distances.erase(it); 

  return norm;
}

void Qjets::Cluster(fastjet::ClusterSequence & cs){
  ComputeDCut(cs);
  ComputeAllDistances(cs.jets());
  while (!_distances.empty()){   
    double dmin = ComputeMinimumDistance(); 
    double norm = ComputeNormalization(dmin);
  
    // sometimes if the rigidity is too large the norm is zero.  make sure this is not the case
    if(_distances.size() == 0)
      break;
    assert(norm > 0.);

    // Now compute a random number between 0 and 1 and find the corresponding measure
    double rand = Rand();
    double sum = 0.;
    //std::cout << rand << std::endl;
    for(list<jet_distance>::iterator it = _distances.begin(); it != _distances.end(); it++){
      /*
	We need to be careful about dmin == 0 which happens for collinear jets.
	(Thanks go to Nhan V Tran for finding this bug)
      */
      if(dmin == 0.){
	if((*it).dij == 0.)
	  sum += 1./norm;
      } else
	sum += exp(-_rigidity*((*it).dij-dmin)/dmin)/norm;
      assert(!isnan((float)sum));
     
      if(sum > rand){
	if(!Prune((*it),cs)){
	  _merged_jets.push_back((*it).j1);
	  _merged_jets.push_back((*it).j2);
	  int new_jet;
	  cs.plugin_record_ij_recombination((*it).j1, (*it).j2, 1., new_jet);
	  assert(JetUnmerged(new_jet));
	  ComputeNewDistanceMeasures(cs,new_jet);
	} else {
	  double j1pt = cs.jets()[(*it).j1].perp();
	  double j2pt = cs.jets()[(*it).j2].perp();
	  if(j1pt>j2pt){
	    _merged_jets.push_back((*it).j2);
	    cs.plugin_record_iB_recombination((*it).j2, 1.);
	  } else {
	    _merged_jets.push_back((*it).j1);
	    cs.plugin_record_iB_recombination((*it).j1, 1.);
	  }
	}
	break;
      }           
    }
  }
  // merge remaining jets with beam
  int num_merged_final(0);
  for(unsigned int i = 0 ; i < cs.jets().size(); i++)
    if(JetUnmerged(i)){
      num_merged_final++;
      cs.plugin_record_iB_recombination(i,1.);
    }
  assert(num_merged_final < 2);
}

bool Qjets::Prune(jet_distance& jd,fastjet::ClusterSequence & cs){
  double pt1 = cs.jets()[jd.j1].perp();
  double pt2 = cs.jets()[jd.j2].perp();
  fastjet::PseudoJet sum_jet = cs.jets()[jd.j1]+cs.jets()[jd.j2];
  double sum_pt = sum_jet.perp();
  double z = min(pt1,pt2)/sum_pt;
  double d = sqrt(cs.jets()[jd.j1].plain_distance(cs.jets()[jd.j2]));
  return (d > _dcut) && (z < _zcut);
}

void Qjets::ComputeAllDistances(const vector<fastjet::PseudoJet>& inp){
  for(unsigned int i = 0 ; i < inp.size()-1; i++){
    // jet-jet distances
    for(unsigned int j = i+1 ; j < inp.size(); j++){
      jet_distance jd;
      jd.j1 = i;
      jd.j2 = j;
      if(jd.j1 != jd.j2){
	jd.dij = d_ij(inp[i],inp[j]);
	_distances.push_back(jd);
      }
    }    
  }
}

double Qjets::d_ij(const fastjet::PseudoJet& v1,const  fastjet::PseudoJet& v2){
  double p1 = v1.perp();
  double p2 = v2.perp();
  double ret = pow(min(p1,p2),_exp_min) * pow(max(p1,p2),_exp_max) * v1.squared_distance(v2);
  assert(!isnan((float)ret));
  return ret;
}

double Qjets::Rand(){
  double ret = 0.;
  if(_rand_seed_set)
    ret = rand_r(&_seed)/(double)RAND_MAX;
  else 
    ret = rand()/(double)RAND_MAX;
  return ret;
}
