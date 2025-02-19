#include "CommonTools/Statistics/interface/SequentialPartitionGenerator.h"

SequentialPartitionGenerator::SequentialPartitionGenerator( int n, int k, int pmin ) :
  the_n(n), the_k(k), the_pmin(pmin), the_pmax(n) {}

SequentialPartitionGenerator::SequentialPartitionGenerator(int n, int k, int pmin, int pmax) :
  the_n(n), the_k(k), the_pmin(pmin), the_pmax(pmax) {}

SequentialPartitionGenerator::Partition SequentialPartitionGenerator::next_partition()
{
  bool done=next_part(the_part);
  if (done)
  {
    return the_part;
  };
  SequentialPartitionGenerator::Partition empty;
  return empty;
}

bool SequentialPartitionGenerator::first_part(
    SequentialPartitionGenerator::Partition & p, int k, int n, int pmin, int pmax ) const
{
  n_first++;
  bool done=false;
  switch (k) {
  case 1:
    p[0]=std::min(pmax,n);
    return (n<=pmax && p[0]>=pmin);
  case 2:
    for (int i=std::min(pmax,n-1);i>=pmin;i--) {
      if ((done=(i>=n-i && n-i>=pmin))) {p[0]=i;p[1]=n-i;}
      return done;
    }
  default:
    Partition pp(p.begin()+1,p.end());
    for (int i=std::min(pmax,n-k+1);i>=pmin;i--) {
      p[0]=i;
      done=this->first_part(pp,k-1,n-p[0],pmin,p[0]);
      if (done) {copy(pp.begin(),pp.end(),p.begin()+1);}
      return done;
    }
  }
  return done;
}

bool SequentialPartitionGenerator::next_part( 
    SequentialPartitionGenerator::Partition & p ) const
{
  n_next++;
  bool done=false;
  int k=p.size();
  switch (k) {
    case 0:  // empty partition: first call to next_part, initialize
      p.insert(p.begin(),the_k,0);
      return this->first_part(p,the_k,the_n,the_pmin,the_pmax);
    case 1:
      return false;
    default:
      int n=0;
      for (int i=0;i<k;i++) n=n+p[i];
      SequentialPartitionGenerator::Partition pp(p.begin()+1,p.end());
      done = (pp.size()>1 ? this->next_part(pp) : false);
      if (done)
      {
        copy(pp.begin(),pp.end(),p.begin()+1);
      } else {
        done = (p[0]==1 ? false :
                this->first_part(pp,k-1,n-p[0]+1,the_pmin,p[0]-1));
        if (done) { --p[0];copy(pp.begin(),pp.end(),p.begin()+1); }
      };
      return done;
  };
  return done;
}
