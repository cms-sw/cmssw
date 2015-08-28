#ifndef __SelfSubtractVoronoiAlgorithm_h__
#define __SelfSubtractVoronoiAlgorithm_h__

#include "GenericVoronoiAlgorithm.h"

class SelfSubtractVoronoiAlgorithm : public GenericVoronoiAlgorithm {
protected:
	double _self_subtract_antikt_distance;
	double _self_subtract_exclusion_perp_min;
	double _self_subtract_exclusion_radius;
private:
	void unsubtracted_momentum(void);
	void self_subtract_momentum(
		const std::vector<bool> &exclusion_density = std::vector<bool>(),
		const std::vector<bool> &exclusion_flow = std::vector<bool>());
	void self_subtract_exclusion(
		std::vector<bool> &exclusion_density,
		std::vector<bool> &exclusion_flow,
		const bool fake_reject,
		const double antikt_distance,
		const double exclusion_perp_min,
		const double exclusion_radius,
		const bool exclusion_by_constituent = false);
	void subtract_momentum(void);
public:
	SelfSubtractVoronoiAlgorithm(
		const double self_subtract_antikt_distance = 0.2,
		const double self_subtract_exclusion_perp_min = 25.0,
		const double self_subtract_exclusion_radius = 0.4,
		const std::pair<double, double> equalization_threshold =
		std::pair<double, double>(5.0, 35.0),
		const bool remove_nonpositive = true);
	~SelfSubtractVoronoiAlgorithm(void);
};


#endif
