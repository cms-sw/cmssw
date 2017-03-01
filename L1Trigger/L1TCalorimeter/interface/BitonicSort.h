#ifndef BitonicSort_h
#define BitonicSort_h

#include <cstdint>
#include <vector>


enum sort_direction {up, down};

// DECLARE!
template <typename T>
void BitonicSort( sort_direction aDir,
		  typename std::vector<T>::iterator & aDataStart,
		  typename std::vector<T>::iterator & aDataEnd
		  );
template <typename T>
void BitonicMerge( sort_direction aDir,
		   typename std::vector<T>::iterator & aDataStart,
		   typename std::vector<T>::iterator & aDataEnd
		   );
//DEFINE!


//SORT
template <typename T>
void BitonicSort( sort_direction aDir,
		  typename std::vector<T>::iterator & aDataStart,
		  typename std::vector<T>::iterator & aDataEnd
		  )
{
  uint32_t lSize( aDataEnd - aDataStart );
  if( lSize > 1 ){
    typename std::vector<T>::iterator lMidpoint( aDataStart+(lSize>>1) );
    if ( aDir == down )
      {
	BitonicSort<T> ( up, aDataStart , lMidpoint );
	BitonicSort<T> ( down, lMidpoint , aDataEnd );
      }else{
      BitonicSort<T> ( down, aDataStart , lMidpoint );
      BitonicSort<T> ( up, lMidpoint , aDataEnd );
    }
    BitonicMerge<T> (aDir, aDataStart , aDataEnd );
  }
}

//MERGE
template <typename T>
void BitonicMerge( sort_direction aDir,
		   typename std::vector<T>::iterator & aDataStart,
		   typename std::vector<T>::iterator & aDataEnd
		   )
{
  uint32_t lSize( aDataEnd - aDataStart );
  if( lSize > 1 ){
    
    uint32_t lPower2(1);
    while (lPower2<lSize) lPower2<<=1;
    
    typename std::vector<T>::iterator lMidpoint( aDataStart + (lPower2>>1) );
    typename std::vector<T>::iterator lFirst( aDataStart );
    typename std::vector<T>::iterator lSecond( lMidpoint );
    
    for( ; lSecond != aDataEnd ; ++lFirst , ++lSecond ){
      if( ( (*lFirst) > (*lSecond) ) == (aDir == up) ) {
	std::swap( *lFirst , *lSecond );
      }
    }
    
    BitonicMerge<T> ( aDir, aDataStart , lMidpoint );
    BitonicMerge<T> ( aDir, lMidpoint , aDataEnd );
  }
}

#endif
