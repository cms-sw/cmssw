#ifndef SequentialCombinationGenerator_H
#define SequentialCombinationGenerator_H

#include "CommonTools/Statistics/interface/SequentialPartitionGenerator.h"
#include <vector>
#include <list>
#include <algorithm>
#include <numeric>
#include <functional>
#include <iostream>

/** 
 * \class SequentialCombinationGenerator
 * Class to compute all distinct Combinations of a collection 'data' of objects
 * of type 'T'.  A Combination is a set of collections, each collection
 * containing one or more objects, with any object in 'data' assigned to
 * exactly one collection.
 */


template<class T> class SequentialCombinationGenerator {
public:
  typedef SequentialPartitionGenerator::Partition Partition;
  typedef std::vector<T> Collection;
  typedef std::vector<Collection> Combination;
  typedef std::vector<int> Vecint;

  SequentialCombinationGenerator ( Partition & part ) :
      the_part ( part ), the_k ( part.size() ),
      the_n ( accumulate ( part.begin(), part.end(), 0 ) )
  {
    sort(the_part.begin(),the_part.end());
    the_comb.reserve(the_n);
  };

  /** Create combinations obtained by dividing 'coll' 
   *  according to partition the_part defined by the constructor. 
   */
  Combination next_combination ( Collection& coll )
  {
    dbg=false;
    Combination comb;
    Vecint newcomb=next_combi(the_comb,the_n,the_part);
    the_comb=newcomb;
    if (newcomb.empty()) {
      Combination empty;
      return empty;
    }
    int i=0;
    for (int j=0;j<the_k;j++) 
    {
      Collection temp;
      for (int l=0;l<the_part[j];l++) 
      {
        temp.push_back(coll[the_comb[i]-1]);
        i++;
      }
      comb.push_back(temp);
    }
    return comb;
  };

private:
  Vecint next_combi( Vecint & cold, int n, const Partition & p )
  {
    Vecint empty;
    if (cold.empty()) { // first entry, initialize
      cold.reserve(n);
      for (int i=0;i<n;cold.push_back(++i));
      return cold;
    }
    int k=p.size();
    if (k==1) return empty;
    Vecint cnew(cold);
    int n1=n-p[0];
    Vecint cold1(cold.begin()+p[0],cold.end());
    Vecint cnew1(n1);
    Partition p1(p.begin()+1,p.end());
    cnew1=next_combi(cold1,n1,p1);
    if (!cnew1.empty()) {
      copy(cnew1.begin(),cnew1.end(),cnew.begin()+p[0]);
      return cnew;
    }
    Vecint cold2(cold.begin(),cold.begin()+p[0]);
    Vecint cnew2(p[0]);
    sort(cold.begin(),cold.end());
    sort(cold2.begin(),cold2.end());
    cnew2=next_subset(cold,cold2);
    if (cnew2.empty()) return empty;
    copy(cnew2.begin(),cnew2.begin()+p[0],cnew.begin());
    Vecint ss(n1);
    set_difference(cold.begin(),cold.end(),
                   cnew2.begin(),cnew2.end(),ss.begin());
    int ip=p[0];
    for (int i=1;i<k;i++) {
      if (p[i]!=p[i-1]) {
        copy(ss.begin(),ss.end(),&cnew[ip]);
        return cnew;
      }
      int mincnew2=cnew2[0];
      if (ss[n1-1]<mincnew2) return empty;
      Vecint::iterator j1=find_if(ss.begin(),ss.end(), [&](auto c) { return c > mincnew2;});
      if (ss.end()-j1 < p[i]) return empty;
      Vecint sss(j1,ss.end());
      for (int j=0;j<p[i];j++){cnew[ip+j]=cnew2[j]=sss[j];}
      int n2=ss.size()-cnew2.size();
      if (n2==0) return cnew;
      Vecint s(n2);
      set_difference(ss.begin(),ss.end(),cnew2.begin(),cnew2.end(),s.begin());
      ss=s;
      ip+=p[i];
    }
   
   return empty; 
 
  };

  Vecint next_subset(const Vecint& _g, const Vecint& _c)
  {
    Vecint g = _g;
    Vecint c = _c;
    Vecint empty;
    int n=g.size();
    int k=c.size();
    typename Vecint::iterator ind;
    for (int i=k-1;i>=0;i--) {
      if (c[i]<g[n-k+i]) {
        
	
//	ind=find(&g[0],&g[n-k+i],c[i])+1;
        
	Vecint::iterator g2 = g.begin();
	advance(g2,n-k+i);  
	
	ind=find(g.begin(),g2,c[i]);
	
	ind++;
	
	
	
	copy(ind,ind+k-i,&c[i]);
        return c;
      }
    }
    return empty;
  };

  void vecprint(const Vecint & v) const
  {
    int n=v.size();
    for (int i=0;i<n;std::cout << v[i++]);
    std::cout << std::endl;
  };

private:
  int the_n;
  int the_k;
  Partition the_part;
  mutable Vecint the_comb;
  mutable bool dbg;
};

#endif
