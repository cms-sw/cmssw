#ifndef RecoHI_HiTracking_BestVertexComparator_h
#define RecoHI_HiTracking_BestVertexComparator_h

template<typename T>
struct GreaterByTracksSize {
	typedef T first_argument_type;
	typedef T second_argument_type;
	bool operator()( const T & t1, const T & t2 ) const {
		return t1.tracksSize() > t2.tracksSize();
	}
};

template<typename T>
struct LessByNormalizedChi2 {
	typedef T first_argument_type;
	typedef T second_argument_type;
	bool operator()( const T & t1, const T & t2 ) const {
		return t1.normalizeChi2() < t2.normalizedChi2();
	}
};

#endif
