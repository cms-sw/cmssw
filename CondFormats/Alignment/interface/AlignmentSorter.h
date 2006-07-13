#ifndef __CondFormats_Alignment_AlignmentSorter_h
#define __CondFormats_Alignment_AlignmentSorter_h

///
/// A struct to sort Alignments and AlignmentErrors by increasing DetId
///
/// To sort Alignments, do something like: 
/// std::sort( alignments->m_align.begin(), alignments->m_align.end(), 
///            lessAlignmentDetId<AlignTransform>() );
///
template<class T>
struct lessAlignmentDetId : public std::binary_function<T,T,bool>
{

  bool operator()( const T& a, const T& b ) 
  { 
	return a.rawId() < b.rawId(); 
  }

};


#endif
