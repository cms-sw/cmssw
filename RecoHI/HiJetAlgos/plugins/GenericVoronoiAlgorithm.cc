#include <fastjet/PseudoJet.hh>
#include <fastjet/ClusterSequence.hh>

#include "GenericVoronoiAlgorithm.h"
#include "BPMPDInterface.h"

using namespace lp;

#include "DataFormats/Math/interface/normalizedPhi.h"

namespace {

	double angular_range_reduce(const double x)
	{
		if (!std::isfinite(x)) {
			return NAN;
		}

		static const double cody_waite_x_max = 1608.4954386379741381;
		static const double two_pi_0 = 6.2831853071795649157;
		static const double two_pi_1 = 2.1561211432631314669e-14;
		static const double two_pi_2 = 1.1615423895917441336e-27;
		double ret = 0;

		if (x >= -cody_waite_x_max && x <= cody_waite_x_max) {
			static const double inverse_two_pi =
				0.15915494309189534197;
			const double k = rint(x * inverse_two_pi);
			ret = ((x - (k * two_pi_0)) - k * two_pi_1) -
				k * two_pi_2;
		}
		else {
			ret = normalizedPhi(ret);
		}
		if (ret == -M_PI) {
			ret = M_PI;
		}

		return ret;
	}
}

void GenericVoronoiAlgorithm::initialize_geometry(void)
{
	static const size_t ncms_hcal_edge_pseudorapidity = 82 + 1;
	static const double cms_hcal_edge_pseudorapidity[
		ncms_hcal_edge_pseudorapidity] = {
		-5.191, -4.889, -4.716, -4.538, -4.363, -4.191, -4.013,
		-3.839, -3.664, -3.489, -3.314, -3.139, -2.964, -2.853,
		-2.650, -2.500, -2.322, -2.172, -2.043, -1.930, -1.830,
		-1.740, -1.653, -1.566, -1.479, -1.392, -1.305, -1.218,
		-1.131, -1.044, -0.957, -0.879, -0.783, -0.696, -0.609,
		-0.522, -0.435, -0.348, -0.261, -0.174, -0.087,
		 0.000,
		 0.087,  0.174,  0.261,  0.348,  0.435,  0.522,  0.609,
		 0.696,  0.783,  0.879,  0.957,  1.044,  1.131,  1.218,
		 1.305,  1.392,  1.479,  1.566,  1.653,  1.740,  1.830,
		 1.930,  2.043,  2.172,  2.322,  2.500,  2.650,  2.853,
		 2.964,  3.139,  3.314,  3.489,  3.664,  3.839,  4.013,
		 4.191,  4.363,  4.538,  4.716,  4.889,  5.191
	};

	_cms_hcal_edge_pseudorapidity = std::vector<double>(
		cms_hcal_edge_pseudorapidity,
		cms_hcal_edge_pseudorapidity +
		ncms_hcal_edge_pseudorapidity);

	static const size_t ncms_ecal_edge_pseudorapidity = 344 + 1;

	for (size_t i = 0; i < ncms_ecal_edge_pseudorapidity; i++) {
		_cms_ecal_edge_pseudorapidity.push_back(
			i * (2 * 2.9928 /
				 (ncms_ecal_edge_pseudorapidity - 1)) -
			2.9928);
	}
}

void GenericVoronoiAlgorithm::allocate(void)
{
	_perp_fourier = new boost::multi_array<double, 4>(
		boost::extents[_edge_pseudorapidity.size() - 1]
		[nreduced_particle_flow_id][nfourier][2]);
}

void GenericVoronoiAlgorithm::deallocate(void)
{
	if (_perp_fourier != NULL) {
		delete _perp_fourier;
	}
}

void GenericVoronoiAlgorithm::event_fourier(void)
{
	std::fill(_perp_fourier->data(),
			  _perp_fourier->data() +
			  _perp_fourier->num_elements(),
			  0);

	for (std::vector<particle_t>::const_iterator iterator =
			 _event.begin();
		 iterator != _event.end(); iterator++) {
		const unsigned int reduced_id =
			iterator->reduced_particle_flow_id;

		for (size_t k = 1; k < _edge_pseudorapidity.size();
			 k++) {
			if (iterator->momentum.Eta() >=
				_edge_pseudorapidity[k - 1] &&
				iterator->momentum.Eta() <
				_edge_pseudorapidity[k]) {
				const double azimuth =
					iterator->momentum.Phi();

				for (size_t l = 0; l < nfourier; l++) {
					(*_perp_fourier)[k - 1][reduced_id]
						[l][0] +=
						iterator->momentum.Pt() *
						cos(l * azimuth);
					(*_perp_fourier)[k - 1][reduced_id]
						[l][1] +=
						iterator->momentum.Pt() *
						sin(l * azimuth);
				}
			}
		}
	}
}

void GenericVoronoiAlgorithm::feature_extract(void)
{
	const size_t nfeature = 2 * nfourier - 1;

	_feature.resize(nfeature);

	// Scale factor to get 95% of the coefficient below 1.0
	// (where however one order of magnitude tolerance is
	// acceptable). This is valid for nfourier < 18 (where
	// interference behavior with the HF geometry starts to
	// appear)

	std::vector<double> scale(nfourier, 1.0 / 200.0);

	if (nfourier >= 1) {
		scale[0] = 1.0 / 5400.0;
	}
	if (nfourier >= 2) {
		scale[1] = 1.0 / 130.0;
	}
	if (nfourier >= 3) {
		scale[2] = 1.0 / 220.0;
	}

	const size_t index_edge_end =
		_edge_pseudorapidity.size() - 2;

	_feature[0] = 0;
	for (size_t j = 0; j < nreduced_particle_flow_id; j++) {
	  _feature[0] += scale[0] *
	    ((*_perp_fourier)[0             ][j][0][0] +
	     (*_perp_fourier)[index_edge_end][j][0][0]);
	}
	
	for (size_t k = 1; k < nfourier; k++) {
	  _feature[2 * k - 1] = 0;
	  for (size_t j = 0; j < nreduced_particle_flow_id; j++) {
	    _feature[2 * k - 1] += scale[k] *
	      ((*_perp_fourier)[0             ][j][k][0] +
		 (*_perp_fourier)[index_edge_end][j][k][0]);
	    }
	  _feature[2 * k] = 0;
	  for (size_t j = 0; j < nreduced_particle_flow_id; j++) {
	    _feature[2 * k] += scale[k] *
	      ((*_perp_fourier)[0             ][j][k][1] +
		 (*_perp_fourier)[index_edge_end][j][k][1]);
	    }
	}


#if 0
	const double event_plane = atan2(_feature[4], _feature[3]);
	const double v2 =
		sqrt(_feature[3] * _feature[3] +
			 _feature[4] * _feature[4]) / _feature[0];
#endif
}

void GenericVoronoiAlgorithm::voronoi_area_incident(void)
{
	// Make the Voronoi diagram

	voronoi_diagram_t diagram;

	// Reverse Voronoi face lookup
#ifdef HAVE_SPARSEHASH
	// The "empty" or default value of the hash table
	const voronoi_diagram_t::Face face_empty;
	google::dense_hash_map<voronoi_diagram_t::Face_handle,
		size_t, hash<voronoi_diagram_t::Face_handle> >
		face_index;

	face_index.set_empty_key(face_empty);
#else // HAVE_SPARSEHASH
	std::map<voronoi_diagram_t::Face_handle, size_t>
		face_index;
#endif // HAVE_SPARSEHASH

	for (std::vector<particle_t>::const_iterator iterator =
			 _event.begin();
		 iterator != _event.end(); iterator++) {
		// Make two additional replicas with azimuth +/- 2 pi
		// (and use only the middle) to mimick the azimuthal
		// cyclicity
		for (int k = -1; k <= 1; k++) {
			const point_2d_t p(
				iterator->momentum.Eta(),
				iterator->momentum.Phi() +
				k * (2 * M_PI));
			const voronoi_diagram_t::Face_handle handle =
				diagram.insert(p);

			face_index[handle] = iterator - _event.begin();
		}
	}

	// Extract the Voronoi cells as polygon and calculate the
	// area associated with individual particles

	for (std::vector<particle_t>::iterator iterator =
			 _event.begin();
		 iterator != _event.end(); iterator++) {
		const voronoi_diagram_t::Locate_result result =
			diagram.locate(*iterator);
		const voronoi_diagram_t::Face_handle *face =
			boost::get<voronoi_diagram_t::Face_handle>(
				&result);
		double polygon_area;

		if (face != NULL) {
			voronoi_diagram_t::Ccb_halfedge_circulator
				circulator_start = (*face)->outer_ccb();
			bool unbounded = false;
			polygon_t polygon;

			voronoi_diagram_t::Ccb_halfedge_circulator
				circulator = circulator_start;

			// Circle around the edges and extract the polygon
			// vertices
			do {
				if (circulator->has_target()) {
					polygon.push_back(
						circulator->target()->point());
					_event[face_index[*face]].incident.
						insert(
							_event.begin() +
							face_index[circulator->twin()->
									   face()]);
				}
				else {
					unbounded = true;
					break;
				}
			}
			while (++circulator != circulator_start);
			if (unbounded) {
				polygon_area = INFINITY;
			}
			else {
				polygon_area = polygon.area();
			}
		}
		else {
			polygon_area = NAN;
		}
		iterator->area = fabs(polygon_area);
	}
}

void GenericVoronoiAlgorithm::recombine_link(void)
{
	boost::multi_array<double, 2> radial_distance_square(
		boost::extents[_event.size()][_event.size()]);

	for (std::vector<particle_t>::const_iterator
			 iterator_outer = _event.begin();
		 iterator_outer != _event.end(); iterator_outer++) {
		radial_distance_square
			[iterator_outer - _event.begin()]
			[iterator_outer - _event.begin()] = 0;

		for (std::vector<particle_t>::const_iterator
				 iterator_inner = _event.begin();
			 iterator_inner != iterator_outer;
			 iterator_inner++) {
			const double deta = iterator_outer->momentum.Eta() -
				iterator_inner->momentum.Eta();
			const double dphi = angular_range_reduce(
				iterator_outer->momentum.Phi() -
				iterator_inner->momentum.Phi());

			radial_distance_square
				[iterator_outer - _event.begin()]
				[iterator_inner - _event.begin()] =
				deta * deta + dphi * dphi;
			radial_distance_square
				[iterator_inner - _event.begin()]
				[iterator_outer - _event.begin()] =
			radial_distance_square
				[iterator_outer - _event.begin()]
				[iterator_inner - _event.begin()];
		}
	}

	_active.clear();

	for (std::vector<particle_t>::const_iterator
			 iterator_outer = _event.begin();
		 iterator_outer != _event.end(); iterator_outer++) {
		double incident_area_sum = iterator_outer->area;

		for (std::set<std::vector<particle_t>::iterator>::
				 const_iterator iterator_inner =
				 iterator_outer->incident.begin();
			 iterator_inner !=
				 iterator_outer->incident.end();
			 iterator_inner++) {
			incident_area_sum += (*iterator_inner)->area;
		}
		_active.push_back(incident_area_sum < 2.0);
	}

	_recombine.clear();
	_recombine_index = std::vector<std::vector<size_t> >(
		_event.size(), std::vector<size_t>());
	_recombine_unsigned = std::vector<std::vector<size_t> >(
		_event.size(), std::vector<size_t>());
	_recombine_tie.clear();

	// 36 cells corresponds to ~ 3 layers, note that for
	// hexagonal tiling, cell in proximity = 3 * layer *
	// (layer + 1)
	static const size_t npair_max = 36;

	for (size_t i = 0; i < _event.size(); i++) {
		for (size_t j = 0; j < _event.size(); j++) {
			const bool active_i_j = _active[i] && _active[j];
			const size_t incident_count =
				_event[i].incident.count(_event.begin() + j) +
				_event[j].incident.count(_event.begin() + i);

			if (active_i_j &&
				(radial_distance_square[i][j] <
				 _radial_distance_square_max ||
				 incident_count > 0)) {
				_recombine_unsigned[i].push_back(j);
			}
		}

		if (_event[i].momentum_perp_subtracted < 0) {
			std::vector<double> radial_distance_square_list;

			for (std::vector<size_t>::const_iterator iterator =
					 _recombine_unsigned[i].begin();
				 iterator != _recombine_unsigned[i].end();
				 iterator++) {
				const size_t j = *iterator;

				if (_event[j].momentum_perp_subtracted > 0) {
					radial_distance_square_list.push_back(
						radial_distance_square[i][j]);
				}
			}

			double radial_distance_square_max_equalization_cut =
				_radial_distance_square_max;

			if (radial_distance_square_list.size() >= npair_max) {
				std::sort(radial_distance_square_list.begin(),
						  radial_distance_square_list.end());
				radial_distance_square_max_equalization_cut =
					radial_distance_square_list[npair_max - 1];
			}

			for (std::vector<size_t>::const_iterator iterator =
					 _recombine_unsigned[i].begin();
				 iterator != _recombine_unsigned[i].end();
				 iterator++) {
				const size_t j = *iterator;

				if (_event[j].momentum_perp_subtracted > 0 &&
					radial_distance_square[i][j] <
					radial_distance_square_max_equalization_cut) {
					_recombine_index[j].push_back(
						_recombine.size());
					_recombine_index[i].push_back(
						_recombine.size());
					_recombine.push_back(
						std::pair<size_t, size_t>(i, j));
					_recombine_tie.push_back(
						radial_distance_square[i][j] /
						_radial_distance_square_max);
				}
			}
		}
	}
}

void GenericVoronoiAlgorithm::lp_populate(void *lp_problem)
{
	bpmpd_problem_t *p = reinterpret_cast<bpmpd_problem_t *>(lp_problem);

	// The minimax problem is transformed into the LP notation
	// using the cost variable trick:
	//
	// Minimize c
	// Subject to:
	// c + sum_l t_kl + n_k >= 0 for negative cells n_k
	// c - sum_k t_kl + p_l >= 0 for positive cells p_l

	// Common LP mistakes during code development and their
	// CPLEX errors when running CPLEX in data checking mode:
	//
	// Error 1201 (column index ... out of range): Bad column
	// indexing, usually index_column out of bound for the
	// cost variables.
	//
	// Error 1222 (duplicate entry): Forgetting to increment
	// index_row, or index_column out of bound for the cost
	// variables.

	p->set_objective_sense(bpmpd_problem_t::minimize);

	// Rows (RHS of the constraints) of the LP problem

	static const size_t nsector_azimuth = 12;

	// Approximatively 2 pi / nsector_azimuth segmentation of
	// the CMS HCAL granularity

	static const size_t ncms_hcal_edge_pseudorapidity = 19 + 1;
	static const double cms_hcal_edge_pseudorapidity[
		ncms_hcal_edge_pseudorapidity] = {
		-5.191, -4.538, -4.013,
		-3.489, -2.853, -2.322, -1.830, -1.305, -0.783, -0.261,
		 0.261,  0.783,  1.305,  1.830,  2.322,  2.853,  3.489,
		 4.013,  4.538,  5.191
	};

	size_t nedge_pseudorapidity;
	const double *edge_pseudorapidity;

	nedge_pseudorapidity = ncms_hcal_edge_pseudorapidity;
	edge_pseudorapidity = cms_hcal_edge_pseudorapidity;

	const size_t nsuperblock = (nedge_pseudorapidity - 2) *
		nsector_azimuth;

	size_t index_row = 0;
	for (size_t index_pseudorapidity = 0;
		 index_pseudorapidity < nedge_pseudorapidity - 2;
		 index_pseudorapidity++) {
		for (size_t index_azimuth = 0;
			 index_azimuth < nsector_azimuth - 1;
			 index_azimuth++) {
			const size_t index_column =
				index_pseudorapidity * nsector_azimuth +
				index_azimuth;
			p->push_back_row(
				bpmpd_problem_t::greater_equal, 0);
			p->push_back_coefficient(
				index_row, index_column, 1);
			p->push_back_coefficient(
				index_row, nsuperblock + index_column, -1);
			index_row++;
			p->push_back_row(
				bpmpd_problem_t::greater_equal, 0);
			p->push_back_coefficient(
				index_row, index_column, 1);
			p->push_back_coefficient(
				index_row, nsuperblock + index_column + 1, -1);
			index_row++;
			p->push_back_row(
				bpmpd_problem_t::greater_equal, 0);
			p->push_back_coefficient(
				index_row, index_column, 1);
			p->push_back_coefficient(
				index_row,
				nsuperblock + index_column + nsector_azimuth, -1);
			index_row++;
			p->push_back_row(
				bpmpd_problem_t::greater_equal, 0);
			p->push_back_coefficient(
				index_row, index_column, 1);
			p->push_back_coefficient(
				index_row,
				nsuperblock + index_column + nsector_azimuth + 1,
				-1);
			index_row++;
		}
		const size_t index_column =
			index_pseudorapidity * nsector_azimuth +
			nsector_azimuth - 1;
		p->push_back_row(
			bpmpd_problem_t::greater_equal, 0);
		p->push_back_coefficient(
			index_row, index_column, 1);
		p->push_back_coefficient(
			index_row, nsuperblock + index_column, -1);
		index_row++;
		p->push_back_row(
			bpmpd_problem_t::greater_equal, 0);
		p->push_back_coefficient(
			index_row, index_column, 1);
		p->push_back_coefficient(
			index_row,
			nsuperblock + index_column - (nsector_azimuth - 1),
			-1);
		index_row++;
		p->push_back_row(
			bpmpd_problem_t::greater_equal, 0);
		p->push_back_coefficient(
			index_row, index_column, 1);
		p->push_back_coefficient(
			index_row,
			nsuperblock + index_column + nsector_azimuth, -1);
		index_row++;
		p->push_back_row(
			bpmpd_problem_t::greater_equal, 0);
		p->push_back_coefficient(
			index_row, index_column, 1);
		p->push_back_coefficient(
			index_row,
			nsuperblock + index_column + nsector_azimuth -
			(nsector_azimuth - 1),
			-1);
		index_row++;
	}

	const size_t nstaggered_block =
		(nedge_pseudorapidity - 1) * nsector_azimuth;
	const size_t nblock = nsuperblock + 2 * nstaggered_block;

	_nblock_subtract = std::vector<size_t>(_event.size(), 0);

	std::vector<size_t>
		positive_index(_event.size(), _event.size());
	size_t positive_count = 0;

	for (std::vector<particle_t>::const_iterator iterator =
			 _event.begin();
		 iterator != _event.end(); iterator++) {
		if (iterator->momentum_perp_subtracted >= 0) {
			positive_index[iterator - _event.begin()] =
				positive_count;
			positive_count++;
		}
	}

	_ncost = nblock + positive_count;

	const double sum_unequalized_0 = _equalization_threshold.first;
	const double sum_unequalized_1 = (2.0 / 3.0) * _equalization_threshold.first + (1.0 / 3.0) * _equalization_threshold.second;
	const double sum_unequalized_2 = (1.0 / 3.0) * _equalization_threshold.first + (2.0 / 3.0) * _equalization_threshold.second;
	const double sum_unequalized_3 = _equalization_threshold.second;

	std::vector<particle_t>::const_iterator
		iterator_particle = _event.begin();
	std::vector<bool>::const_iterator iterator_active =
		_active.begin();
	std::vector<std::vector<size_t> >::const_iterator
		iterator_recombine_index_outer =
		_recombine_index.begin();
	std::vector<std::vector<size_t> >::const_iterator
		iterator_recombine_unsigned_outer =
		_recombine_unsigned.begin();
	size_t index_column_max = _ncost - 1;
	for (; iterator_particle != _event.end();
		 iterator_particle++, iterator_active++,
			 iterator_recombine_index_outer++,
			 iterator_recombine_unsigned_outer++) {
		if (*iterator_active) {
			int index_pseudorapidity = -1;

/////////////////////////////////////////////////////////////////////
			for (size_t i = 1; i < nedge_pseudorapidity; i++) {
				if (iterator_particle->momentum.Eta() >= edge_pseudorapidity[i - 1] &&
					iterator_particle->momentum.Eta() < edge_pseudorapidity[i]) {
					index_pseudorapidity = i - 1;
				}
			}

			const int index_azimuth = floor(
				(iterator_particle->momentum.Phi() + M_PI) *
				((nsector_azimuth >> 1) / M_PI));

			if (index_pseudorapidity != -1) {
				// p_i - sum t - u = c_i
				// or: c_i + u + sum_t = p_i
				// n_i + sum t - u <= 0
				// or: u - sum_t >= n_i

				// Inequality RHS
				p->push_back_row(
					iterator_particle->momentum_perp_subtracted >= 0 ?
					bpmpd_problem_t::equal :
					bpmpd_problem_t::greater_equal,
					iterator_particle->momentum_perp_subtracted);

				// Energy transfer coefficients t_kl
				const double sign = iterator_particle->momentum_perp_subtracted >= 0 ? 1 : -1;
				const size_t index_column_block_subtract =
					nsuperblock +
					(nedge_pseudorapidity - 1) * nsector_azimuth +
					index_pseudorapidity * nsector_azimuth +
					index_azimuth;

				_nblock_subtract[iterator_particle - _event.begin()] =
					index_column_block_subtract;

				if (iterator_particle->momentum_perp_subtracted >= 0) {
					const size_t index_column_cost =
						nblock + positive_index[iterator_particle - _event.begin()];

					p->push_back_coefficient(
						index_row, index_column_cost, 1);
					index_column_max =
						std::max(index_column_max, index_column_cost);
				}
				p->push_back_coefficient(
					index_row, index_column_block_subtract, 1);
				index_column_max =
					std::max(index_column_max, index_column_block_subtract);

				for (std::vector<size_t>::const_iterator
						 iterator_recombine_index_inner =
						 iterator_recombine_index_outer->begin();
					 iterator_recombine_index_inner !=
						 iterator_recombine_index_outer->end();
					 iterator_recombine_index_inner++) {
					const size_t index_column =
						*iterator_recombine_index_inner +
						_ncost;

					p->push_back_coefficient(
						index_row, index_column, sign);
					index_column_max =
						std::max(index_column_max, index_column);
				}
				index_row++;

				const size_t index_column_block =
					nsuperblock +
					index_pseudorapidity * nsector_azimuth +
					index_azimuth;

				// sum_R c_i - o_i >= -d
				// or: d + sum_R c_i >= o_i
				// sum_R c_i - o_i <= d
				// or: d - sum_R c_i >= -o_i

				double sum_unequalized;

				sum_unequalized = 0;
				for (std::vector<size_t>::const_iterator
						 iterator_recombine_unsigned_inner =
						 iterator_recombine_unsigned_outer->begin();
					 iterator_recombine_unsigned_inner !=
						 iterator_recombine_unsigned_outer->end();
					 iterator_recombine_unsigned_inner++) {
					sum_unequalized +=
						_event[*iterator_recombine_unsigned_inner].momentum_perp_subtracted;
				}
				sum_unequalized = std::max(0.0, sum_unequalized);

				if (sum_unequalized >= sum_unequalized_3 ||
					(sum_unequalized >= sum_unequalized_2 &&
					 (iterator_particle - _event.begin()) % 2 == 0) ||
					(sum_unequalized >= sum_unequalized_1 &&
					 (iterator_particle - _event.begin()) % 4 == 0) ||
					(sum_unequalized >= sum_unequalized_0 &&
					 (iterator_particle - _event.begin()) % 8 == 0)) {

				const double weight = sum_unequalized *
					std::min(1.0, std::max(1e-3,
						iterator_particle->area));

				if (weight > 0) {
					p->push_back_row(
						bpmpd_problem_t::greater_equal,
						sum_unequalized);

					p->push_back_coefficient(
						index_row, index_column_block, 1.0 / weight);

					for (std::vector<size_t>::const_iterator
							 iterator_recombine_unsigned_inner =
							 iterator_recombine_unsigned_outer->begin();
						 iterator_recombine_unsigned_inner !=
							 iterator_recombine_unsigned_outer->end();
						 iterator_recombine_unsigned_inner++) {
						if (_event[*iterator_recombine_unsigned_inner].momentum_perp_subtracted >= 0) {
							const size_t index_column_cost =
								nblock +
								positive_index[*iterator_recombine_unsigned_inner];

							p->push_back_coefficient(
								index_row, index_column_cost, 1);
							index_column_max =
								std::max(index_column_max, index_column_cost);
						}
					}
					index_row++;

					p->push_back_row(
						bpmpd_problem_t::greater_equal,
						-sum_unequalized);

					p->push_back_coefficient(
						index_row, index_column_block, _positive_bound_scale / weight);

					for (std::vector<size_t>::const_iterator iterator_recombine_unsigned_inner = iterator_recombine_unsigned_outer->begin();
						 iterator_recombine_unsigned_inner != iterator_recombine_unsigned_outer->end();
						 iterator_recombine_unsigned_inner++) {
						if (_event[*iterator_recombine_unsigned_inner].momentum_perp_subtracted >= 0) {
							const size_t index_column_cost =
								nblock +
								positive_index[*iterator_recombine_unsigned_inner];

							p->push_back_coefficient(
								index_row, index_column_cost, -1);
							index_column_max =
								std::max(index_column_max, index_column_cost);
						}
					}
					index_row++;
				}

				}

			}
		}
	}

	// Epsilon that breaks the degeneracy, in the same units
	// as the pT of the event (i.e. GeV)
	static const double epsilon_degeneracy = 1e-2;

	// Columns (variables and the objective coefficients) of
	// the LP problem
	//
	// Cost variables (objective coefficient 1)
	for (size_t i = 0; i < nsuperblock; i++) {
		p->push_back_column(
			1, 0, bpmpd_problem_t::infinity);
	}
	for (size_t i = nsuperblock; i < nsuperblock + nstaggered_block; i++) {
		p->push_back_column(
			0, 0, bpmpd_problem_t::infinity);
	}
	for (size_t i = nsuperblock + nstaggered_block; i < nsuperblock + 2 * nstaggered_block; i++) {
		p->push_back_column(
			0, 0, bpmpd_problem_t::infinity);
	}
	for (size_t i = nsuperblock + 2 * nstaggered_block; i < _ncost; i++) {
		p->push_back_column(
			0, 0, bpmpd_problem_t::infinity);
	}
	//fprintf(stderr, "%s:%d: %lu %lu\n", __FILE__, __LINE__, index_column_max, recombine_tie.size());
	// Energy transfer coefficients t_kl (objective
	// coefficient 0 + epsilon)
	for (size_t i = _ncost; i <= index_column_max; i++) {
		p->push_back_column(
			epsilon_degeneracy * _recombine_tie[i - _ncost],
			0, bpmpd_problem_t::infinity);
	}
}
void GenericVoronoiAlgorithm::equalize(void)
{
	bpmpd_problem_t lp_problem;

	recombine_link();
	lp_populate(&lp_problem);
	lp_problem.optimize();

	int solution_status;
	double objective_value;
	std::vector<double> x;
	std::vector<double> pi;

	lp_problem.solve(solution_status, objective_value,
					  x, pi);

	for (size_t k = _ncost; k < x.size(); k++) {
		if (_event[_recombine[k - _ncost].first].
			momentum_perp_subtracted < 0 &&
			_event[_recombine[k - _ncost].second].
			momentum_perp_subtracted >= 0 && x[k] >= 0) {
		_event[_recombine[k - _ncost].first].
			momentum_perp_subtracted += x[k];
		_event[_recombine[k - _ncost].second].
			momentum_perp_subtracted -= x[k];
		}
	}
	for (size_t k = 0; k < _event.size(); k++) {
		if (_nblock_subtract[k] != 0 &&
			x[_nblock_subtract[k]] >= 0) {
			_event[k].momentum_perp_subtracted -=
				x[_nblock_subtract[k]];
		}
	}
}
void GenericVoronoiAlgorithm::remove_nonpositive(void)
{
	for (std::vector<particle_t>::iterator iterator =
			 _event.begin();
		 iterator != _event.end(); iterator++) {
		iterator->momentum_perp_subtracted = std::max(
			0.0, iterator->momentum_perp_subtracted);
	}
}
void GenericVoronoiAlgorithm::subtract_if_necessary(void)
{
	if (!_subtracted) {
		event_fourier();
		feature_extract();
		voronoi_area_incident();
		subtract_momentum();
		if (_remove_nonpositive) {
			equalize();
			remove_nonpositive();
		}
		_subtracted = true;
	}
}

GenericVoronoiAlgorithm::GenericVoronoiAlgorithm(
	const double dr_max,
	const std::pair<double, double> equalization_threshold,
	const bool remove_nonpositive)
	: _equalization_threshold(equalization_threshold),
	  _remove_nonpositive(remove_nonpositive),
	  _radial_distance_square_max(dr_max * dr_max),
	  _positive_bound_scale(0.2),
	  _subtracted(false), _perp_fourier(NULL)
{
	initialize_geometry();

	static const size_t nedge_pseudorapidity = 15 + 1;
	static const double edge_pseudorapidity[nedge_pseudorapidity] = {
-5.191, -2.650, -2.043, -1.740, -1.479, -1.131, -0.783, -0.522, 0.522, 0.783, 1.131, 1.479, 1.740, 2.043, 2.650, 5.191
	};

	_edge_pseudorapidity = std::vector<double>(
edge_pseudorapidity,
edge_pseudorapidity + nedge_pseudorapidity);
	allocate();
}

GenericVoronoiAlgorithm::~GenericVoronoiAlgorithm(void)
{
	deallocate();
}

/**
 * Add a new unsubtracted particle to the current event
 *
 * @param[in]	perp	transverse momentum
 * @param[in]	pseudorapidity	pseudorapidity
 * @param[in]	azimuth	azimuth
 * @param[in]	reduced_particle_flow_id	reduced particle
 * flow ID, between 0 and 2 (inclusive)
 */
void GenericVoronoiAlgorithm::push_back_particle(
	const double perp, const double pseudorapidity,
	const double azimuth,
	const unsigned int reduced_particle_flow_id)
{
	math::PtEtaPhiELorentzVector p(perp, pseudorapidity, azimuth, NAN);

	p.SetE(p.P());
	_event.push_back(particle_t(p, reduced_particle_flow_id));
}

/**
 * Clears the list of unsubtracted particles
 */
void GenericVoronoiAlgorithm::clear(void)
{
	_event.clear();
	_subtracted = false;
}

/**
 * Returns the transverse momenta of the subtracted particles
 *
 * @return	vector of transverse momenta
 */
std::vector<double> GenericVoronoiAlgorithm::subtracted_equalized_perp(void)
{
	subtract_if_necessary();

	std::vector<double> ret;

	for (std::vector<particle_t>::const_iterator iterator =
			 _event.begin();
		 iterator != _event.end(); iterator++) {
		ret.push_back(iterator->momentum_perp_subtracted);
	}

	return ret;
}

std::vector<double> GenericVoronoiAlgorithm::subtracted_unequalized_perp(void)
{
	subtract_if_necessary();

	std::vector<double> ret;

	for (std::vector<particle_t>::const_iterator iterator =
			_event.begin();
		iterator != _event.end(); iterator++) {
		ret.push_back(iterator->momentum_perp_subtracted_unequalized);
	}

	return ret;
}

/**
 * Returns the area in the Voronoi diagram diagram occupied by
 * a given particle
 *
 * @return	vector of area
 */
std::vector<double> GenericVoronoiAlgorithm::particle_area(void)
{
	subtract_if_necessary();

	std::vector<double> ret;

	for (std::vector<particle_t>::const_iterator iterator =
			 _event.begin();
		 iterator != _event.end(); iterator++) {
		ret.push_back(iterator->area);
	}

	return ret;
}

/**
 * Returns the incident particles in the Delaunay diagram
 * (particles that has a given particle as the nearest
 * neighbor)
 *
 * @return	vector of sets of incident particles
 * indices, using the original indexing
 */
std::vector<std::set<size_t> > GenericVoronoiAlgorithm::particle_incident(void)
{
	subtract_if_necessary();

	std::vector<std::set<size_t> > ret;

	for (std::vector<particle_t>::const_iterator
			 iterator_outer = _event.begin();
		 iterator_outer != _event.end(); iterator_outer++) {
		std::set<size_t> e;

		for (std::set<std::vector<particle_t>::iterator>::
				 const_iterator iterator_inner =
				 iterator_outer->incident.begin();
			 iterator_inner != iterator_outer->incident.begin();
			 iterator_inner++) {
			e.insert(*iterator_inner - _event.begin());
		}
		ret.push_back(e);
	}

	return ret;
}

std::vector<double> GenericVoronoiAlgorithm::perp_fourier(void)
{
	subtract_if_necessary();

	return std::vector<double>(
		_perp_fourier->data(),
		_perp_fourier->data() +
		_perp_fourier->num_elements());
}

size_t GenericVoronoiAlgorithm::nedge_pseudorapidity(void) const
{
	return _edge_pseudorapidity.size();
}
