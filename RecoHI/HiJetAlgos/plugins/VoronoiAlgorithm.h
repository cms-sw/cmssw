#ifndef __VoronoiAlgorithm_h__
#define __VoronoiAlgorithm_h__

#include "GenericVoronoiAlgorithm.h"

class VoronoiAlgorithm : public GenericVoronoiAlgorithm {
public:
	static const size_t nreduced_particle_flow_id = 3;
	static const size_t nfourier = 5;
protected:
	// calibrations
	const UECalibration *ue;
	const bool exclude_v1;
	const unsigned int max_vn;
	const bool diagonal_vn;
private:
	void event_fourier(void);
	void feature_extract(void);
	void voronoi_area_incident(void);
	void subtract_momentum(void);
	void recombine_link(void);
	void lp_populate(void *lp_problem);
	void subtract_if_necessary(void);
public:
	VoronoiAlgorithm(
		const UECalibration *ue,
		const double dr_max,
		const bool exclude_v1,
		const int max_vn,
		const bool diagonal_vn,
		const std::pair<double, double> equalization_threshold =
		std::pair<double, double>(5.0, 35.0),
		const bool remove_nonpositive = true);
	~VoronoiAlgorithm(void);
	size_t nedge_pseudorapidity(void) const;
};

#endif
