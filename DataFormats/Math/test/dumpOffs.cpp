
   template<unsigned int D>
   struct RowOffsets {
      inline RowOffsets() {
         int v[D];
         v[0]=0;
         for (unsigned int i=1; i<D; ++i)
            v[i]=v[i-1]+i;
         for (unsigned int i=0; i<D; ++i) { 
            for (unsigned int j=0; j<=i; ++j)
               fOff[i*D+j] = v[i]+j; 
            for (unsigned int j=i+1; j<D; ++j)
               fOff[i*D+j] = v[j]+i ;
         }
      }
      inline int operator()(unsigned int i, unsigned int j) const { return fOff[i*D+j]; }
      inline int apply(unsigned int i) const { return fOff[i]; }
      int fOff[D*D];
   };

#include<iostream>
template<unsigned int D>
void dump( RowOffsets<D> const & ro ) {
  std::cout << "{ ";
  for (int i=0; i!=D*D-1; ++i)
    std::cout << ro.apply(i) << ",";
  std::cout << ro.apply(D*D-1) << " }," << std::endl;
}

int  main() {
  RowOffsets<15> ro;
  dump(ro);
  return 0;
}
