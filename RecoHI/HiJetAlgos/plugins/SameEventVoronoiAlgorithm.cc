#include <fastjet/PseudoJet.hh>
#include <fastjet/ClusterSequence.hh>

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "SameEventVoronoiAlgorithm.h"

namespace {

	// Same as the result from ParticleTowerProducer::beginJob when
	// useHF_ = true
	static const double etaedge[42] = {
		0, 0.087, 0.174, 0.261, 0.348, 0.435, 0.522, 0.609, 0.696,
		0.783, 0.87, 0.957, 1.044, 1.131, 1.218, 1.305, 1.392, 1.479,
		1.566, 1.653, 1.74, 1.83, 1.93, 2.043, 2.172, 2.322, 2.5,
		2.65, 2.868, 3, 3.139, 3.314, 3.489, 3.664, 3.839, 4.013,
		4.191, 4.363, 4.538, 4.716, 4.889, 5.191
	};

	// Taken from
	// FastSimulation/CalorimeterProperties/src/HCALProperties.cc Note
	// this returns an abs(ieta)
	int eta2ieta(double eta)
	{
		// binary search in the array of towers eta edges

		int size = sizeof(etaedge) / sizeof(double);
		// if(!useHF_) size = 30;

		if(fabs(eta)>etaedge[size-1]) return -1;

		double x = fabs(eta);
		int curr = size / 2;
		int step = size / 4;
		int iter;
		int prevdir = 0; 
		int actudir = 0; 

		for (iter = 0; iter < size ; iter++) {

			if( curr >= size || curr < 1 )
				edm::LogError("SameEventVoronoiAlgorithm") << "eta2ieta - wrong current index = "
						  << curr << " !!!" << std::endl;

			if ((x <= etaedge[curr]) && (x > etaedge[curr-1])) break;
			prevdir = actudir;
			if(x > etaedge[curr]) {actudir =  1;}
			else {actudir = -1;}
			if(prevdir * actudir < 0) { if(step > 1) step /= 2;}
			curr += actudir * step;
			if(curr > size) curr = size;
			else { if(curr < 1) {curr = 1;}}

			/*
			  std::cout << " HCALProperties::eta2ieta  end of iter." << iter 
			  << " curr, etaedge[curr-1], etaedge[curr] = "
			  << curr << " " << etaedge[curr-1] << " " << etaedge[curr] << std::endl;
			*/
    
		}

		/*
		  std::cout << " HCALProperties::eta2ieta  for input x = " << x 
		  << "  found index = " << curr-1
          << std::endl;
		*/
  
		return curr;
	}

	int phi2iphi(double phi, int ieta)
	{
  
		if(phi<0) phi += 2.*M_PI;
		else if(phi> 2.*M_PI) phi -= 2.*M_PI;
  
		int iphi = (int) ceil(phi/2.0/M_PI*72.);
		// take into account larger granularity in endcap (x2) and at the end of the HF (x4)
		if(abs(ieta)>20){
			if(abs(ieta)<40) iphi -= (iphi+1)%2;
			else {
				iphi -= (iphi+1)%4;
				if(iphi==-1) iphi=71;
			}
		}
  
		return iphi;

	}
}


void SameEventVoronoiAlgorithm::unsubtracted_momentum(void)
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

void SameEventVoronoiAlgorithm::same_event_subtract_momentum(
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
	if (density_0_sum > 0) {
		std::transform(density_0.begin(), density_0.end(),
					   density_2.begin(),
					   std::bind1st(std::multiplies<double>(),
									density_2_sum / density_0_sum));
	}
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

void SameEventVoronoiAlgorithm::same_event_exclusion(
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
				const int int_pseudorapidity_positive =
					eta2ieta(iterator_constituent->pseudorapidity());
				if (int_pseudorapidity_positive >= 0) {
					const int int_pseudorapidity =
						(iterator_constituent->pseudorapidity() < 0 ? -1 : 1) *
						int_pseudorapidity_positive;
					const int int_azimuth =
						phi2iphi(iterator_constituent->phi_std(),
								 int_pseudorapidity);

					pseudotower[
						std::pair<int, int>(int_pseudorapidity, int_azimuth)] +=
						_event[iterator_constituent->user_index()].
						momentum_perp_subtracted_unequalized;
				}
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
			if (pseudotower.size() > 0) {
				et_mean /= pseudotower.size();
			}
			if (et_mean > 0) {
				jet_excluded &= (et_max >= _same_event_fake_reject_et_max &&
								 et_max / et_mean >=
								 _same_event_fake_reject_et_max_over_mean);
			}
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

void SameEventVoronoiAlgorithm::subtract_momentum(void)
{
	// ATLAS Collab., Phys. Lett. B 719, 220-241 (2013) adapted to PF
	// and without track jets (not possible with the current
	// VirtualJetProducer)

	// Initialize for reconstruction of seed jets

	unsubtracted_momentum();

	std::vector<bool> exclusion_density;
	std::vector<bool> exclusion_flow;

	// Reconstructing the seed jets with R =
	// _same_event_antikt_distance (0.2 for ATLAS)
	same_event_exclusion(
		exclusion_density, exclusion_flow,
		true,	// Apply ATLAS E_T^max/<E_T> fake rejection
		_same_event_antikt_distance,
		0,		// Exclude all seed jets passing fake rejection (min pT = 0)
		0,		// Not needed if exclusion_by_constituent = true
		true	// Exclude by constituent (not by radial distance)
	);
	// Subtract excluding all seed jets
	same_event_subtract_momentum(exclusion_density);

	// ATLAS 2nd step (note we still do not subtract the constituents
	// before clustering, the jet definition is including the UE, same
	// as ATLAS; jet subtraction only occurs when comparing against
	// _same_event_exclusion_perp_min)

	same_event_exclusion(
		exclusion_density, exclusion_flow,
		false,
		_same_event_antikt_distance,
		_same_event_exclusion_perp_min,
		_same_event_exclusion_radius);
	// Subtract excluding the updated seed jets >=
	// _same_event_exclusion_perp_min, now including Delta eta
	// strips
	same_event_subtract_momentum(exclusion_density, exclusion_flow);
}

SameEventVoronoiAlgorithm::SameEventVoronoiAlgorithm(
	const double antikt_distance,
	const double exclusion_perp_min,
	const double exclusion_radius,
	const double fake_reject_et_max,
	const double fake_reject_et_max_over_mean,
	const std::pair<double, double> equalization_threshold,
	const bool remove_nonpositive)
	: GenericVoronoiAlgorithm(exclusion_radius, equalization_threshold,
							  remove_nonpositive),
	  _same_event_antikt_distance(antikt_distance),
	  _same_event_exclusion_perp_min(exclusion_perp_min),
	  _same_event_exclusion_radius(exclusion_radius),
	  _same_event_fake_reject_et_max(fake_reject_et_max),
	  _same_event_fake_reject_et_max_over_mean(fake_reject_et_max_over_mean)
{
	allocate();
}
