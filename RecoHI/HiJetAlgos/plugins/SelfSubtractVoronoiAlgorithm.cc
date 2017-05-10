#include <fastjet/PseudoJet.hh>
#include <fastjet/ClusterSequence.hh>

#include "SelfSubtractVoronoiAlgorithm.h"

void SelfSubtractVoronoiAlgorithm::unsubtracted_momentum(void)
{
	for (std::vector<particle_t>::iterator iterator =
			 _event.begin();
		 iterator != _event.end(); iterator++) {
		iterator->momentum_perp_subtracted =
			iterator->momentum.Pt();
		iterator->momentum_perp_subtracted_unequalized =
			iterator->momentum_perp_subtracted;
	}
}

void SelfSubtractVoronoiAlgorithm::self_subtract_momentum(
	const std::vector<bool> &exclusion_density,
	const std::vector<bool> &exclusion_flow)
{
	// event_plane_2 = 2 * Psi_2 (where -pi < Psi_2 < pi)
	const double event_plane_2 = atan2(_feature[4], _feature[3]);
	std::vector<int> interpolation_index;

	for (size_t i = 0; i < _event.size(); i++) {
		interpolation_index.push_back(-1);

		for (size_t l = 1;
			 l < _cms_hcal_edge_pseudorapidity.size(); l++) {
			if (_event[i].momentum.Eta() >=
				_cms_hcal_edge_pseudorapidity[l - 1] &&
				_event[i].momentum.Eta() <
				_cms_hcal_edge_pseudorapidity[l]) {
				interpolation_index.back() = l - 1;
				break;
			}
		}
	}

	std::vector<double>
		density_0(_cms_hcal_edge_pseudorapidity.size() - 1, 0);
	std::vector<double>
		density_2(_cms_hcal_edge_pseudorapidity.size() - 1, 0);
	std::vector<double>
		area_density_0(_cms_hcal_edge_pseudorapidity.size() - 1, 0);
	std::vector<double>
		area_density_2(_cms_hcal_edge_pseudorapidity.size() - 1, 0);

	for (size_t i = 0; i < _event.size(); i++) {
		if (interpolation_index[i] != -1) {
			if (exclusion_density.empty() || !exclusion_density[i]) {
				density_0[interpolation_index[i]] +=
					_event[i].momentum.Pt();
				area_density_0[interpolation_index[i]] +=
					_event[i].area;
			}
			if (exclusion_flow.empty() || !exclusion_flow[i]) {
				density_2[interpolation_index[i]] +=
					_event[i].momentum.Pt() *
					cos(2 * _event[i].momentum.Phi() - event_plane_2);
				area_density_2[interpolation_index[i]] +=
					_event[i].area;
			}
		}
	}

	// v2 averaging

	double density_0_sum = 0;
	double density_2_sum = 0;

	for (size_t i = 0; i < area_density_0.size(); i++) {
		if (area_density_0[i] > 0 && area_density_2[i] > 0) {
			density_0_sum += density_0[i];
			density_2_sum += density_2[i];
		}
	}
	std::transform(density_0.begin(), density_0.end(), density_2.begin(),
				   std::bind1st(std::multiplies<double>(),
								density_2_sum / density_0_sum));
	for (size_t i = 0; i < area_density_0.size(); i++) {
		if (area_density_0[i] > 0) {
			density_0[i] /= area_density_0[i];
		}
	}
	for (size_t i = 0; i < area_density_2.size(); i++) {
		if (area_density_2[i] > 0) {
			density_2[i] /= area_density_2[i];
		}
	}

	for (size_t i = 0; i < _event.size(); i++) {
		if (interpolation_index[i] != -1) {
			const double density =
				density_0[interpolation_index[i]]
				+ 2 * density_2[interpolation_index[i]] *
				cos(2 * _event[i].momentum.Phi() - event_plane_2)
				;

			if (std::isfinite(_event[i].area) && density >= 0) {
				_event[i].momentum_perp_subtracted =
					_event[i].momentum.Pt() -
					density * _event[i].area;
			}
			else {
				_event[i].momentum_perp_subtracted =
					_event[i].momentum.Pt();
			}
		}
		else {
			_event[i].momentum_perp_subtracted =
				_event[i].momentum.Pt();
		}
		_event[i].momentum_perp_subtracted_unequalized =
			_event[i].momentum_perp_subtracted;
	}
}

void SelfSubtractVoronoiAlgorithm::self_subtract_exclusion(
	std::vector<bool> &exclusion_density,
	std::vector<bool> &exclusion_flow,
	const bool fake_reject,
	const double antikt_distance,
	const double exclusion_perp_min,
	const double exclusion_radius,
	const bool exclusion_by_constituent)
{
	std::vector<fastjet::PseudoJet> pseudojet;

	for (std::vector<particle_t>::const_iterator iterator =
			 _event.begin();
		 iterator != _event.end(); iterator++) {
		pseudojet.push_back(fastjet::PseudoJet(
			iterator->momentum.px(),
			iterator->momentum.py(),
			iterator->momentum.pz(),
			iterator->momentum.energy()));
		pseudojet.back().set_user_index(
			iterator - _event.begin());
	}

	fastjet::JetDefinition
		jet_definition(fastjet::antikt_algorithm,
					   antikt_distance);
	fastjet::ClusterSequence
		cluster_sequence(pseudojet, jet_definition);
	std::vector<fastjet::PseudoJet> jet =
		cluster_sequence.inclusive_jets();

	exclusion_density = std::vector<bool>(_event.size(), false);
	exclusion_flow = std::vector<bool>(_event.size(), false);

	for (std::vector<fastjet::PseudoJet>::const_iterator
			 iterator_jet = jet.begin();
		 iterator_jet != jet.end(); iterator_jet++) {
		std::vector<fastjet::PseudoJet> constituent =
			cluster_sequence.constituents(*iterator_jet);
		double perp_resummed = 0;

		for (std::vector<fastjet::PseudoJet>::const_iterator
				 iterator_constituent = constituent.begin();
			 iterator_constituent != constituent.end();
			 iterator_constituent++) {
			perp_resummed +=
				_event[iterator_constituent->user_index()].
				momentum_perp_subtracted_unequalized;
		}

		bool jet_excluded = perp_resummed >= exclusion_perp_min;

		if (fake_reject) {
			// ATLAS E_T^max/<E_T> fake rejection using 0.1x0.1
			// pseudotower, as in Phys. Lett. B 719 (2013) 222 (left
			// column).

			std::map<std::pair<int, int>, double> pseudotower;

			for (std::vector<fastjet::PseudoJet>::const_iterator
				 iterator_constituent = constituent.begin();
			 iterator_constituent != constituent.end();
			 iterator_constituent++) {
				const int int_pseudorapidity =
					floor(iterator_constituent->pseudorapidity() *
						  10.0);
				const int int_azimuth =
					floor((iterator_constituent->phi_std() + M_PI) *
						  (32.0 / M_PI));

				pseudotower[
					std::pair<int, int>(int_pseudorapidity, int_azimuth)] +=
					_event[iterator_constituent->user_index()].
					momentum_perp_subtracted_unequalized;
			}

			double et_max = 0;
			double et_mean = 0;

			for (std::map<std::pair<int, int>, double>::const_iterator
					 iterator_pseudotower = pseudotower.begin();
				 iterator_pseudotower != pseudotower.end();
				 iterator_pseudotower++) {
				et_max = std::max(et_max, iterator_pseudotower->second);
				et_mean += iterator_pseudotower->second;
			}
			et_mean /= pseudotower.size();

			jet_excluded &= (et_max >= 3 && et_max / et_mean >= 4);
		}

		if (jet_excluded) {
			if (exclusion_by_constituent) {
				for (std::vector<fastjet::PseudoJet>::const_iterator
						 iterator_constituent = constituent.begin();
					 iterator_constituent != constituent.end();
					 iterator_constituent++) {
					const size_t index =
						iterator_constituent->user_index();

					exclusion_density[index] = true;
				}
			}
			else {
				for (std::vector<fastjet::PseudoJet>::const_iterator
						 iterator_pseudojet = pseudojet.begin();
					 iterator_pseudojet != pseudojet.end();
					 iterator_pseudojet++) {
					const size_t index =
						iterator_pseudojet->user_index();

					if (iterator_jet->squared_distance(
						*iterator_pseudojet) <
						exclusion_radius * exclusion_radius) {
						exclusion_density[index] = true;
					}
					if (fabs(iterator_pseudojet->pseudorapidity() -
							 iterator_jet->pseudorapidity()) <
						exclusion_radius) {
						exclusion_flow[index] = true;
					}
				}
			}
		}
	}
}

void SelfSubtractVoronoiAlgorithm::subtract_momentum(void)
{
	// ATLAS Collab., Phys. Lett. B 719, 220-241 (2013) adapted to PF
	// and without track jets (not possible with the current
	// VirtualJetProducer)

	// Initialize for reconstruction of seed jets

	unsubtracted_momentum();

	std::vector<bool> exclusion_density;
	std::vector<bool> exclusion_flow;

	// Reconstructing the seed jets with R =
	// _self_subtract_antikt_distance (0.2 for ATLAS)
	self_subtract_exclusion(
		exclusion_density, exclusion_flow,
		true,	// Apply ATLAS E_T^max/<E_T> fake rejection
		_self_subtract_antikt_distance,
		0,		// Exclude all seed jets passing fake rejection (min pT = 0)
		0,		// Not needed if exclusion_by_constituent = true
		true	// Exclude by constituent (not by radial distance)
	);
	// Subtract excluding all seed jets
	self_subtract_momentum(exclusion_density);

	// ATLAS 2nd step (note we still do not subtract the constituents
	// before clustering, the jet definition is including the UE, same
	// as ATLAS; jet subtraction only occurs when comparing against
	// _self_subtract_exclusion_perp_min)

	self_subtract_exclusion(
		exclusion_density, exclusion_flow,
		false,
		_self_subtract_antikt_distance,
		_self_subtract_exclusion_perp_min,
		_self_subtract_exclusion_radius);
	// Subtract excluding the updated seed jets >=
	// _self_subtract_exclusion_perp_min, now including Delta eta
	// strips
	self_subtract_momentum(exclusion_density, exclusion_flow);
}

SelfSubtractVoronoiAlgorithm::SelfSubtractVoronoiAlgorithm(
	const double antikt_distance,
	const double exclusion_perp_min,
	const double exclusion_radius,
	const std::pair<double, double> equalization_threshold,
	const bool remove_nonpositive)
	: GenericVoronoiAlgorithm(exclusion_radius, equalization_threshold,
							  remove_nonpositive),
	  _self_subtract_antikt_distance(antikt_distance),
	  _self_subtract_exclusion_perp_min(exclusion_perp_min),
	  _self_subtract_exclusion_radius(exclusion_radius)
{
	allocate();
}
