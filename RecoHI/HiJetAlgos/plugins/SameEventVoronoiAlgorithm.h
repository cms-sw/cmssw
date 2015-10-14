#ifndef __SameEventVoronoiAlgorithm_h__
#define __SameEventVoronoiAlgorithm_h__

#include "GenericVoronoiAlgorithm.h"

class SameEventVoronoiAlgorithm : public GenericVoronoiAlgorithm {
protected:
	double _same_event_antikt_distance;
	double _same_event_exclusion_perp_min;
	double _same_event_exclusion_radius;
	double _same_event_fake_reject_et_max;
	double _same_event_fake_reject_et_max_over_mean;
private:
	void unsubtracted_momentum(void);
	void same_event_subtract_momentum(
		const std::vector<bool> &exclusion_density = std::vector<bool>(),
		const std::vector<bool> &exclusion_flow = std::vector<bool>());
	void same_event_exclusion(
		std::vector<bool> &exclusion_density,
		std::vector<bool> &exclusion_flow,
		const bool fake_reject,
		const double antikt_distance,
		const double exclusion_perp_min,
		const double exclusion_radius,
		const bool exclusion_by_constituent = false);
	void subtract_momentum(void);
public:
	SameEventVoronoiAlgorithm(
		const double same_event_antikt_distance = 0.2,
		const double same_event_exclusion_perp_min = 25.0,
		const double same_event_exclusion_radius = 0.4,
		const double same_event_fake_reject_et_max = 3,
		const double same_event_fake_reject_et_max_over_mean = 4,
		const std::pair<double, double> equalization_threshold =
		std::pair<double, double>(5.0, 35.0),
		const bool remove_nonpositive = true);
	~SameEventVoronoiAlgorithm(void);
};


#endif
