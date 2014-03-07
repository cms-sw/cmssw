#include "RecoJets/JetAlgorithms/interface/Qjets.h"

using namespace std;

void Qjets::SetRandSeed(unsigned int seed){
  _rand_seed_set = true;
  _seed = seed;
}

bool Qjets::JetUnmerged(int num) const{
  return _merged_jets.find(num) == _merged_jets.end();
}

bool Qjets::JetsUnmerged(const JetDistance& jd) const{
  return JetUnmerged(jd.j1) && JetUnmerged(jd.j2);
}

JetDistance Qjets::GetNextDistance(){
  vector< pair<JetDistance, double> > popped_distances;
  double norm(0.);
  JetDistance ret;
  ret.j1 = -1;
  ret.j2 = -1;
  ret.dij = -1.;
  bool dmin_set(false);
  double dmin(0.);

  while(!_distances.empty()){
    JetDistance dist = _distances.top();
    _distances.pop();
    if(JetsUnmerged(dist)){
      if(!dmin_set){
	dmin = dist.dij;
	dmin_set = true;
      }
      double weight = exp(-_rigidity * (dist.dij-dmin) /dmin);
      popped_distances.push_back(make_pair(dist,weight));        
      norm += weight; 
      if(weight/norm < _truncation_fctr)
	break;            
    }
  }
  
  double rand(Rand()), tot_weight(0.);
  for(vector<pair<JetDistance, double> >::iterator it = popped_distances.begin(); it != popped_distances.end(); it++){
    tot_weight += (*it).second/norm;
    if(tot_weight >= rand){
      ret = (*it).first;
      break;
    }
  }
  
  // repopulate in reverse (maybe quicker?)
  for(vector<pair<JetDistance, double> >::reverse_iterator it = popped_distances.rbegin(); it != popped_distances.rend(); it++)
    if(JetsUnmerged((*it).first))
      _distances.push((*it).first);

  return ret;
}

void Qjets::Cluster(fastjet::ClusterSequence & cs){
  computeDCut(cs);

  // Populate all the distances
  ComputeAllDistances(cs.jets());
  JetDistance jd = GetNextDistance();

  while(!_distances.empty() && jd.dij != -1.){
    if(!Prune(jd,cs)){
      //      _merged_jets.push_back(jd.j1);
      //      _merged_jets.push_back(jd.j2);
      _merged_jets[jd.j1] = true;
      _merged_jets[jd.j2] = true;

      int new_jet=-1;
      cs.plugin_record_ij_recombination(jd.j1, jd.j2, 1., new_jet);
      if(!JetUnmerged(new_jet))
	edm::LogError("QJets Clustering") << "Problem with FastJet::plugin_record_ij_recombination";
      ComputeNewDistanceMeasures(cs,new_jet);
    } else {
      double j1pt = cs.jets()[jd.j1].perp();
      double j2pt = cs.jets()[jd.j2].perp();
      if(j1pt>j2pt){
	//	_merged_jets.push_back(jd.j2);
	_merged_jets[jd.j2] = true;
	cs.plugin_record_iB_recombination(jd.j2, 1.);
      } else {
	//	_merged_jets.push_back(jd.j1);
	_merged_jets[jd.j1] = true;
	cs.plugin_record_iB_recombination(jd.j1, 1.);
      }
    }
    jd = GetNextDistance();
  } 

  // merge remaining jets with beam
  int num_merged_final(0);
  for(unsigned int i = 0 ; i < cs.jets().size(); i++)
    if(JetUnmerged(i)){
      num_merged_final++;
      cs.plugin_record_iB_recombination(i,1.);
    }
  
  if(!(num_merged_final < 2))
    edm::LogError("QJets Clustering") << "More than 1 jet remains after reclustering";
}

void Qjets::computeDCut(fastjet::ClusterSequence & cs){
  // assume all jets in cs form a single jet.  compute mass and pt
  fastjet::PseudoJet sum(0.,0.,0.,0.);
  for(vector<fastjet::PseudoJet>::const_iterator it = cs.jets().begin(); it != cs.jets().end(); it++)
    sum += (*it);
  _dcut = 2. * _dcut_fctr * sum.m()/sum.perp(); 
}

bool Qjets::Prune(JetDistance& jd,fastjet::ClusterSequence & cs){
  double pt1 = cs.jets()[jd.j1].perp();
  double pt2 = cs.jets()[jd.j2].perp();
  fastjet::PseudoJet sum_jet = cs.jets()[jd.j1]+cs.jets()[jd.j2];
  double sum_pt = sum_jet.perp();
  double z = min(pt1,pt2)/sum_pt;
  double d2 = cs.jets()[jd.j1].plain_distance(cs.jets()[jd.j2]);
  return (d2 > _dcut*_dcut) && (z < _zcut);
}

void Qjets::ComputeAllDistances(const vector<fastjet::PseudoJet>& inp){
  for(unsigned int i = 0 ; i < inp.size()-1; i++){
    // jet-jet distances
    for(unsigned int j = i+1 ; j < inp.size(); j++){
      JetDistance jd;
      jd.j1 = i;
      jd.j2 = j;
      if(jd.j1 != jd.j2){
	jd.dij = d_ij(inp[i],inp[j]);
	_distances.push(jd);
      }
    }    
  }
}

void Qjets::ComputeNewDistanceMeasures(fastjet::ClusterSequence & cs, unsigned int new_jet){
  // jet-jet distances
  for(unsigned int i = 0; i < cs.jets().size(); i++)
    if(JetUnmerged(i) && i != new_jet){
      JetDistance jd;
      jd.j1 = new_jet;
      jd.j2 = i;
      jd.dij = d_ij(cs.jets()[jd.j1], cs.jets()[jd.j2]);
      _distances.push(jd);
    }
}

double Qjets::d_ij(const fastjet::PseudoJet& v1,const  fastjet::PseudoJet& v2) const{
  double p1 = v1.perp();
  double p2 = v2.perp();
  double ret = pow(min(p1,p2),_exp_min) * pow(max(p1,p2),_exp_max) * v1.squared_distance(v2);
  if(edm::isNotFinite(ret))
    edm::LogError("QJets Clustering") << "d_ij is not finite";
  return ret; 
}

double Qjets::Rand(){
  double ret = 0.;
  //if(_rand_seed_set)
  //  ret = rand_r(&_seed)/(double)RAND_MAX;
  //else 
  //ret = rand()/(double)RAND_MAX;
  ret = _rnEngine->flat();
  return ret;
}
