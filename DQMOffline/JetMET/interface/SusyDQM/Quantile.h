#ifndef QUANTILE_H
#define QUANTILE_H

#include "TH1.h"
#include <vector>
#include <iostream>

struct Quantile {
  typedef std::pair<double,double> pair;
  typedef std::vector<pair> array;
  
  pair operator()(const double frac) const {return fromHead(frac);}
  pair operator[](const double frac) const {return fromTail(frac);}

  Quantile(const TH1* h) : 
    N( 1 + h->GetNbinsX()),
    Total(h->Integral(0,N))
  { for(int i=0;i<N; i++) {
      const double H = h->GetBinContent(i)   + (head.size()?head.back().second:0);
      const double T = h->GetBinContent(N-i) + (tail.size()?tail.back().second:0);  
      if(H) head.push_back( pair( h->GetBinWidth(i) + h->GetBinLowEdge(i) , H));
      if(T) tail.push_back( pair(                     h->GetBinLowEdge(N-i),T)); 
    }
  }
  
  pair fromHead(const double frac) const {return calculateQ(frac,true);}
  pair fromTail(const double frac) const {return calculateQ(frac,false);}

private:

  pair calculateQ(const double frac, const bool fromHead) const {
    const double f = frac<0.5 ? frac : 1-frac ;
    array::const_iterator 
      begin( ( (frac<0.5) == fromHead ) ?  head.begin() : tail.begin()), 
      end(   ( (frac<0.5) == fromHead ) ?  head.end()   : tail.end()), 
      bin(begin);

    while( bin->second < f*Total ) bin++;
//dk    if( bin==begin ) return pair(sqrt(-1),0);
    if( bin==begin ) return pair(-1,0);
  
    array::const_iterator 
      binNext( next(bin,end)),
      binPrev( prev(bin,begin)),
      binPPrev( prev(binPrev,begin));

    const double
      DX( binNext->first - binPPrev->first ),
      DY( (binNext->second - binPPrev->second)/Total ),

      dX( bin->first - binPrev->first ),
      dY( (bin->second - binPrev->second)/Total ),

      avgX( ( bin->first + binPrev->first) /2 ),
      avgY( ( bin->second + binPrev->second) /(2*Total) ),

      x_q( avgX + dX/dY * ( f - avgY ) ),
      xerr_q( std::max(fabs(DX/DY),fabs(dX/dY)) * sqrt(f*(1-f)/Total) );
    
    return pair(x_q,xerr_q);
  }
  

  template<class T> T prev(T bin, T begin) const {
    T binPrev = bin;
    while( binPrev > begin && 
	   (binPrev-1)->second == (bin-1)->second ) 
      binPrev--;
    return binPrev;
  }
  
  template<class T> T next(T bin, T end) const {
    T binNext = bin;
    while( binNext<end-1 &&
	   (++binNext)->second == bin->second) 
      ;
    return binNext;
  }

  const int N;
  const double Total;
  array head,tail;
};
#endif
