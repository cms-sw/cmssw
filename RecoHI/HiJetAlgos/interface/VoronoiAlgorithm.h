#ifndef __VoronoiAlgorithm_h__
#define __VoronoiAlgorithm_h__

#include <cmath>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Voronoi_diagram_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_traits_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_policies_2.h>
#include <CGAL/Polygon_2.h>

#include <boost/multi_array.hpp>

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "RecoHI/HiJetAlgos/interface/UECalibration.h"

class VoronoiAlgorithm {
private:
	//
	typedef CGAL::Delaunay_triangulation_2<
	CGAL::Exact_predicates_inexact_constructions_kernel>
	delaunay_triangulation_t;
	typedef delaunay_triangulation_t::Point point_2d_t;
	class particle_t {
	public:
		math::PtEtaPhiELorentzVector momentum;
		unsigned int reduced_particle_flow_id;
		double area;
		double momentum_perp_subtracted;
		std::set<std::vector<particle_t>::iterator> incident;
		particle_t(math::PtEtaPhiELorentzVector p,
				   unsigned int i, double a = NAN,
				   double ps = NAN)
			: momentum(p), reduced_particle_flow_id(i), area(a),
			  momentum_perp_subtracted(ps),
			  incident(std::set<std::vector<particle_t>::
					   iterator>())
		{
		}
		inline operator point_2d_t(void) const
		{
			return point_2d_t(momentum.Eta(), momentum.Phi());
		}
	};
	typedef std::vector<particle_t> event_t;
	// Remaining CGAL classes
	typedef CGAL::Voronoi_diagram_2<
		delaunay_triangulation_t,
		CGAL::Delaunay_triangulation_adaptation_traits_2<
			delaunay_triangulation_t>,
		CGAL::
		Delaunay_triangulation_caching_degeneracy_removal_policy_2<
			delaunay_triangulation_t> > voronoi_diagram_t;
	typedef CGAL::Polygon_2<
		CGAL::Exact_predicates_inexact_constructions_kernel>
	polygon_t;
public:
	static const size_t nreduced_particle_flow_id = 3;
	static const size_t nfourier = 3;
protected:
	std::vector<double> _edge_pseudorapidity;
	std::vector<double> _cms_hcal_edge_pseudorapidity;
	std::vector<double> _cms_ecal_edge_pseudorapidity;
	bool _remove_nonpositive;
	double _radial_distance_square_max;
	double _positive_bound_scale;
	bool _subtracted;
	event_t _event;
	boost::multi_array<double, 4> *_perp_fourier;
	std::vector<double> _feature;
	std::vector<bool> _active;
	std::vector<std::pair<size_t, size_t> > _recombine;
	std::vector<std::vector<size_t> > _recombine_index;
	std::vector<std::vector<size_t> > _recombine_unsigned;
	std::vector<double> _recombine_tie;
	size_t _ncost;
	std::vector<size_t> _nblock_subtract;
	void *_lp_environment;
	void *_lp_problem;
	// calibrations
	UECalibration ue;
private:
	void initialize_geometry(void);
	void allocate(void);
	void deallocate(void);
	void event_fourier(void);
	void feature_extract(void);
	void voronoi_area_incident(void);
	void subtract_momentum(void);
	void recombine_link(void);
	void lp_populate(void *lp_problem);
	void equalize(void);
	void remove_nonpositive(void);
	void subtract_if_necessary(void);
public:
	VoronoiAlgorithm(const double dr_max,
					 const bool remove_nonpositive = true);
	VoronoiAlgorithm(const double dr_max,
					 const bool remove_nonpositive,
					 const std::vector<double> edge_pseudorapidity);
	~VoronoiAlgorithm(void)
	{
	}
<<<<<<< HEAD

	class VoronoiAlgorithm {
	private:
		//
		typedef CGAL::Delaunay_triangulation_2<
			CGAL::Exact_predicates_inexact_constructions_kernel>
		delaunay_triangulation_t;
		typedef delaunay_triangulation_t::Point point_2d_t;
		class particle_t {
		public:
			snowmass_vector_t<double> momentum;
			unsigned int reduced_particle_flow_id;
			double area;
			double momentum_perp_subtracted;
			std::set<std::vector<particle_t>::iterator> incident;
			particle_t(snowmass_vector_t<double> p,
					   unsigned int i, double a = NAN,
					   double ps = NAN)
				: momentum(p), reduced_particle_flow_id(i), area(a),
				  momentum_perp_subtracted(ps),
				  incident(std::set<std::vector<particle_t>::
						   iterator>())
			{
			}
			inline operator point_2d_t(void) const
			{
				return point_2d_t(momentum.pseudorapidity(),
								  momentum.azimuth());
			}
		};
		typedef std::vector<particle_t> event_t;
		// Remaining CGAL classes
		typedef CGAL::Voronoi_diagram_2<
			delaunay_triangulation_t,
			CGAL::Delaunay_triangulation_adaptation_traits_2<
				delaunay_triangulation_t>,
			CGAL::
			Delaunay_triangulation_caching_degeneracy_removal_policy_2<
				delaunay_triangulation_t> > voronoi_diagram_t;
		typedef CGAL::Polygon_2<
			CGAL::Exact_predicates_inexact_constructions_kernel>
		polygon_t;
	public:
		static const size_t nreduced_particle_flow_id = 3;
		static const size_t nfourier = 3;
	protected:
		std::vector<double> _edge_pseudorapidity;
		std::vector<double> _cms_hcal_edge_pseudorapidity;
		std::vector<double> _cms_ecal_edge_pseudorapidity;
		bool _remove_nonpositive;
		double _radial_distance_square_max;
		double _positive_bound_scale;
		bool _subtracted;
		event_t _event;
		boost::multi_array<double, 4> *_perp_fourier;
		std::vector<double> _feature;
		std::vector<bool> _active;
		std::vector<std::pair<size_t, size_t> > _recombine;
		std::vector<std::vector<size_t> > _recombine_index;
		std::vector<std::vector<size_t> > _recombine_unsigned;
		std::vector<double> _recombine_tie;
		size_t _ncost;
		std::vector<size_t> _nblock_subtract;
		bpmpd_environment_t _lp_environment;
		bpmpd_problem_t _lp_problem;
#ifndef STANDALONE
		// calibrations
	        UECalibration* ue;
#endif // STANDALONE
	private:
		void initialize_geometry(void)
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
		void allocate(void)
		{
			_perp_fourier = new boost::multi_array<double, 4>(
				boost::extents[_edge_pseudorapidity.size() - 1]
				[nreduced_particle_flow_id][nfourier][2]);
		}
		void deallocate(void)
		{
			delete _perp_fourier;
		}
		void event_fourier(void)
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
					if (iterator->momentum.pseudorapidity() >=
						_edge_pseudorapidity[k - 1] &&
						iterator->momentum.pseudorapidity() <
						_edge_pseudorapidity[k]) {
						const double azimuth =
							iterator->momentum.azimuth();

						for (size_t l = 0; l < nfourier; l++) {
							(*_perp_fourier)[k - 1][reduced_id]
								[l][0] +=
								iterator->momentum.perp() *
								cos(l * azimuth);
							(*_perp_fourier)[k - 1][reduced_id]
								[l][1] +=
								iterator->momentum.perp() *
								sin(l * azimuth);
						}
					}
				}
			}
		}
		void feature_extract(void)
		{
			const size_t nfeature = 2 * nfourier - 1;

			_feature.resize(nfeature);

			static const double scale[3] = {
				1.0 / 4950, 1.0 / 140, 1.0 / 320
			};

			const size_t index_edge_end =
				_edge_pseudorapidity.size() - 2;

			_feature[0] = scale[0] *
				((*_perp_fourier)[0             ][2][0][0] +
				 (*_perp_fourier)[index_edge_end][2][0][0]);
			for (size_t k = 1; k < nfourier; k++) {
				_feature[2 * k - 1] = scale[k] *
					((*_perp_fourier)[0             ][2][k][0] +
					 (*_perp_fourier)[index_edge_end][2][k][0]);
				_feature[2 * k] = scale[k] *
					((*_perp_fourier)[0             ][2][k][1] +
					 (*_perp_fourier)[index_edge_end][2][k][1]);
			}

#if 0
			const double event_plane = atan2(_feature[4], _feature[3]);
			const double v2 =
				sqrt(_feature[3] * _feature[3] +
					 _feature[4] * _feature[4]) / _feature[0];
#endif
		}
		void voronoi_area_incident(void)
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
						iterator->momentum.pseudorapidity(),
						iterator->momentum.azimuth() +
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
		void subtract_momentum(void)
		{
			for (std::vector<particle_t>::iterator iterator =
					 _event.begin();
				 iterator != _event.end(); iterator++) {
				int predictor_index = -1;
				int interpolation_index = -1;
				double density = 0;
				double pred_0 = 0;

				for (size_t l = 1; l < _edge_pseudorapidity.size(); l++) {
					if (iterator->momentum.pseudorapidity() >=
						_edge_pseudorapidity[l - 1] &&
						iterator->momentum.pseudorapidity() <
						_edge_pseudorapidity[l]) {
						predictor_index = l - 1;
					}
				}

				for (size_t j = 0; j < 3; j++) {
				if (j == 2) {
					// HCAL
					for (size_t l = 1;
						 l < _cms_hcal_edge_pseudorapidity.size(); l++) {
						if (iterator->momentum.pseudorapidity() >=
							_cms_hcal_edge_pseudorapidity[l - 1] &&
							iterator->momentum.pseudorapidity() <
							_cms_hcal_edge_pseudorapidity[l]) {
							interpolation_index = l - 1;
						}
					}
				}
				else {
					// Tracks or ECAL clusters
					for (size_t l = 1;
						 l < _cms_ecal_edge_pseudorapidity.size(); l++) {
						if (iterator->momentum.pseudorapidity() >=
							_cms_ecal_edge_pseudorapidity[l - 1] &&
							iterator->momentum.pseudorapidity() <
							_cms_ecal_edge_pseudorapidity[l]) {
							interpolation_index = l - 1;
						}
					}
				}

				if (predictor_index >= 0 && interpolation_index >= 0) {
					// Calculate the aggregated prediction and
					// interpolation for the pseudorapidity segment

					const double azimuth = iterator->momentum.azimuth();
					const float (*p)[2][46] =
#ifdef STANDALONE
						ue_predictor_pf[j][predictor_index]
#else // STANDALONE
						ue->ue_predictor_pf[j][predictor_index]
#endif // STANDALONE
						;
					double pred = 0;

					for (size_t l = 0; l < 3; l++) {
						for (size_t m = 0; m < 2; m++) {
							float u = p[l][m][0];

							for (size_t n = 0; n < 2 * nfourier - 1; n++) {
								u += (((((((((p[l][m][9 * n + 9]) *
											 _feature[n] +
											 p[l][m][9 * n + 8]) *
											_feature[n] +
											p[l][m][9 * n + 7]) *
										   _feature[n] +
										   p[l][m][9 * n + 6]) *
										  _feature[n] +
										  p[l][m][9 * n + 5]) *
										 _feature[n] +
										 p[l][m][9 * n + 4]) *
										_feature[n] +
										p[l][m][9 * n + 3]) *
									   _feature[n] +
									   p[l][m][9 * n + 2]) *
									  _feature[n] +
									  p[l][m][9 * n + 1]) *
									_feature[n];
							}

							pred += u * (l == 0 ? 1.0 : 2.0) *
								(m == 0 ? cos(l * azimuth) :
								 sin(l * azimuth));
							if (l == 0 && m == 0) {
								pred_0 += u /
									(2.0 * M_PI *
									 (_edge_pseudorapidity[predictor_index + 1] -
									  _edge_pseudorapidity[predictor_index]));
							}
						}
					}

					double interp;

#ifdef STANDALONE
					if (j == 0) {
						interp =
							ue_interpolation_pf0[predictor_index][
								interpolation_index];
					}
					else if (j == 1) {
						interp =
							ue_interpolation_pf1[predictor_index][
								interpolation_index];
					}
					else if (j == 2) {
						interp =
							ue_interpolation_pf2[predictor_index][
								interpolation_index];
					}
#else // STANDALONE
					if (j == 0) {
						interp =
							ue->ue_interpolation_pf0[predictor_index][
								interpolation_index];
					}
					else if (j == 1) {
						interp =
							ue->ue_interpolation_pf1[predictor_index][
								interpolation_index];
					}
					else if (j == 2) {
						interp =
							ue->ue_interpolation_pf2[predictor_index][
								interpolation_index];
					}
#endif // STANDALONE
					// Interpolate down to the finely binned
					// pseudorapidity

					density += pred /
						(2.0 * M_PI *
						 (_edge_pseudorapidity[predictor_index + 1] -
						  _edge_pseudorapidity[predictor_index])) *
						interp;
				}
				}

					if (std::isfinite(iterator->area)) {
						// Subtract the PF candidate by density times
						// Voronoi cell area
						iterator->momentum_perp_subtracted =
							iterator->momentum.perp() -
							density * iterator->area;
					}
					else {
						iterator->momentum_perp_subtracted =
							iterator->momentum.perp();
					}
			}
		}
		void recombine_link(bpmpd_problem_t &_lp_problem)
		{
			boost::multi_array<double, 2> radial_distance_square(
				boost::extents[_event.size()][_event.size()]);

			for (std::vector<particle_t>::const_iterator
					 iterator_outer = _event.begin();
				 iterator_outer != _event.end(); iterator_outer++) {
				for (std::vector<particle_t>::const_iterator
						 iterator_inner = _event.begin();
					 iterator_inner != _event.end();
					 iterator_inner++) {
					radial_distance_square
						[iterator_outer - _event.begin()]
						[iterator_inner - _event.begin()] =
						iterator_outer->momentum.
						radial_distance_square(
							iterator_inner->momentum);
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

			for (size_t i = 0; i < _event.size(); i++) {
				if (_event[i].momentum_perp_subtracted < 0) {
					for (size_t j = 0; j < _event.size(); j++) {
						const bool active_i_j =
							_active[i] && _active[j];
						// We take advantage of std::set::count()
						// returning 0 or 1, and test for a positive
						// sum.
						size_t incident_count =
							_event[i].incident.count(_event.begin() + j) +
							_event[j].incident.count(_event.begin() + i);

						if (_event[j].momentum_perp_subtracted > 0 &&
							active_i_j &&
							(radial_distance_square[i][j] <
							 _radial_distance_square_max ||
							 incident_count > 0)) {
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
			}
		}
		void lp_populate(bpmpd_problem_t &_lp_problem)
		{
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

			_lp_problem.set_objective_sense(bpmpd_problem_t::minimize);

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
					_lp_problem.push_back_row(
						bpmpd_problem_t::greater_equal, 0);
					_lp_problem.push_back_coefficient(
						index_row, index_column, 1);
					_lp_problem.push_back_coefficient(
						index_row, nsuperblock + index_column, -1);
					index_row++;
					_lp_problem.push_back_row(
						bpmpd_problem_t::greater_equal, 0);
					_lp_problem.push_back_coefficient(
						index_row, index_column, 1);
					_lp_problem.push_back_coefficient(
						index_row, nsuperblock + index_column + 1, -1);
					index_row++;
					_lp_problem.push_back_row(
						bpmpd_problem_t::greater_equal, 0);
					_lp_problem.push_back_coefficient(
						index_row, index_column, 1);
					_lp_problem.push_back_coefficient(
						index_row,
						nsuperblock + index_column + nsector_azimuth, -1);
					index_row++;
					_lp_problem.push_back_row(
						bpmpd_problem_t::greater_equal, 0);
					_lp_problem.push_back_coefficient(
						index_row, index_column, 1);
					_lp_problem.push_back_coefficient(
						index_row,
						nsuperblock + index_column + nsector_azimuth + 1,
						-1);
					index_row++;
				}
				const size_t index_column =
					index_pseudorapidity * nsector_azimuth +
					nsector_azimuth - 1;
				_lp_problem.push_back_row(
					bpmpd_problem_t::greater_equal, 0);
				_lp_problem.push_back_coefficient(
					index_row, index_column, 1);
				_lp_problem.push_back_coefficient(
					index_row, nsuperblock + index_column, -1);
				index_row++;
				_lp_problem.push_back_row(
					bpmpd_problem_t::greater_equal, 0);
				_lp_problem.push_back_coefficient(
					index_row, index_column, 1);
				_lp_problem.push_back_coefficient(
					index_row,
					nsuperblock + index_column - (nsector_azimuth - 1),
					-1);
				index_row++;
				_lp_problem.push_back_row(
					bpmpd_problem_t::greater_equal, 0);
				_lp_problem.push_back_coefficient(
					index_row, index_column, 1);
				_lp_problem.push_back_coefficient(
					index_row,
					nsuperblock + index_column + nsector_azimuth, -1);
				index_row++;
				_lp_problem.push_back_row(
					bpmpd_problem_t::greater_equal, 0);
				_lp_problem.push_back_coefficient(
					index_row, index_column, 1);
				_lp_problem.push_back_coefficient(
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
						if (iterator_particle->momentum.pseudorapidity() >= edge_pseudorapidity[i - 1] &&
							iterator_particle->momentum.pseudorapidity() < edge_pseudorapidity[i]) {
							index_pseudorapidity = i - 1;
						}
					}

					const int index_azimuth = floor(
						(iterator_particle->momentum.azimuth() + M_PI) *
						((nsector_azimuth >> 1) / M_PI));

					if (index_pseudorapidity != -1) {
						// p_i - sum t - u = c_i
						// or: c_i + u + sum_t = p_i
						// n_i + sum t - u <= 0
						// or: u - sum_t >= n_i

						// Inequality RHS
						_lp_problem.push_back_row(
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

							_lp_problem.push_back_coefficient(
								index_row, index_column_cost, 1);
							index_column_max =
								std::max(index_column_max, index_column_cost);
						}
						_lp_problem.push_back_coefficient(
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

							_lp_problem.push_back_coefficient(
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

						const double weight = sum_unequalized;

						if (weight > 0) {
							_lp_problem.push_back_row(
								bpmpd_problem_t::greater_equal,
								sum_unequalized);

							_lp_problem.push_back_coefficient(
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

									_lp_problem.push_back_coefficient(
										index_row, index_column_cost, 1);
									index_column_max =
										std::max(index_column_max, index_column_cost);
								}
							}
							index_row++;

							_lp_problem.push_back_row(
								bpmpd_problem_t::greater_equal,
								-sum_unequalized);

							_lp_problem.push_back_coefficient(
								index_row, index_column_block, _positive_bound_scale / weight);

							for (std::vector<size_t>::const_iterator iterator_recombine_unsigned_inner = iterator_recombine_unsigned_outer->begin();
								 iterator_recombine_unsigned_inner != iterator_recombine_unsigned_outer->end();
								 iterator_recombine_unsigned_inner++) {
								if (_event[*iterator_recombine_unsigned_inner].momentum_perp_subtracted >= 0) {
									const size_t index_column_cost =
										nblock +
										positive_index[*iterator_recombine_unsigned_inner];

									_lp_problem.push_back_coefficient(
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

			// Epsilon that breaks the degeneracy, in the same units
			// as the pT of the event (i.e. GeV)
			static const double epsilon_degeneracy = 1e-2;

			// Columns (variables and the objective coefficients) of
			// the LP problem
			//
			// Cost variables (objective coefficient 1)
			for (size_t i = 0; i < nsuperblock; i++) {
				_lp_problem.push_back_column(
					1, 0, bpmpd_problem_t::infinity);
			}
			for (size_t i = nsuperblock; i < nsuperblock + nstaggered_block; i++) {
				_lp_problem.push_back_column(
					0, 0, bpmpd_problem_t::infinity);
			}
			for (size_t i = nsuperblock + nstaggered_block; i < nsuperblock + 2 * nstaggered_block; i++) {
				_lp_problem.push_back_column(
					0, 0, bpmpd_problem_t::infinity);
			}
			for (size_t i = nsuperblock + 2 * nstaggered_block; i < _ncost; i++) {
				_lp_problem.push_back_column(
					0, 0, bpmpd_problem_t::infinity);
			}
			//fprintf(stderr, "%s:%d: %lu %lu\n", __FILE__, __LINE__, index_column_max, recombine_tie.size());
			// Energy transfer coefficients t_kl (objective
			// coefficient 0 + epsilon)
			for (size_t i = _ncost; i <= index_column_max; i++) {
				_lp_problem.push_back_column(
					epsilon_degeneracy * _recombine_tie[i - _ncost],
					0, bpmpd_problem_t::infinity);
			}
		}
		void equalize(void)
		{
			bpmpd_problem_t _lp_problem = _lp_environment.problem();

			recombine_link(_lp_problem);
			lp_populate(_lp_problem);
			_lp_problem.optimize();

			int solution_status;
			double objective_value;
			std::vector<double> x;
			std::vector<double> pi;

			_lp_problem.solve(solution_status, objective_value,
							  x, pi);

			for (size_t k = _ncost; k < x.size(); k++) {
				_event[_recombine[k - _ncost].first].
					momentum_perp_subtracted += x[k];
				_event[_recombine[k - _ncost].second].
					momentum_perp_subtracted -= x[k];
			}
			for (size_t k = 0; k < _event.size(); k++) {
				if (_nblock_subtract[k] != 0) {
					_event[k].momentum_perp_subtracted -=
						x[_nblock_subtract[k]];
				}
			}
		}
		void remove_nonpositive(void)
		{
			for (std::vector<particle_t>::iterator iterator =
					 _event.begin();
				 iterator != _event.end(); iterator++) {
				iterator->momentum_perp_subtracted = std::max(
					0.0, iterator->momentum_perp_subtracted);
			}
		}
		void subtract_if_necessary(void)
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
	public:
		VoronoiAlgorithm(const double dr_max,
				 bool isRealData = true, 
				 const bool remove_nonpositive = true)
			: _remove_nonpositive(remove_nonpositive),
			  _radial_distance_square_max(dr_max * dr_max),
			  _positive_bound_scale(0.2),
		  _subtracted(false),
		  ue(0)
		{
			initialize_geometry();
			ue = new UECalibration(isRealData);
			static const size_t nedge_pseudorapidity = 7 + 1;
			static const double edge_pseudorapidity[nedge_pseudorapidity] = {
				-5.191, -3.0, -1.479, -0.522, 0.522, 1.479, 3.0, 5.191
			};

			_edge_pseudorapidity = std::vector<double>(
				edge_pseudorapidity,
				edge_pseudorapidity + nedge_pseudorapidity);
			allocate();
		}
		VoronoiAlgorithm(const double dr_max,
						 const bool remove_nonpositive,
						 const std::vector<double> edge_pseudorapidity)
			: _edge_pseudorapidity(edge_pseudorapidity),
			  _remove_nonpositive(remove_nonpositive),
			  _radial_distance_square_max(dr_max * dr_max),
			  _positive_bound_scale(0.2),
			  _subtracted(false)
		{
			initialize_geometry();
			allocate();
		}
		~VoronoiAlgorithm(void)
		{
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
		void push_back_particle(
			const double perp, const double pseudorapidity,
			const double azimuth,
			const unsigned int reduced_particle_flow_id)
		{
			snowmass_vector_t<double> p(NAN, perp, pseudorapidity, azimuth);

			p.set_lightlike_time();
			_event.push_back(particle_t(p, reduced_particle_flow_id));
		}
		/**
		 * Clears the list of unsubtracted particles
		 */
		void clear(void)
		{
			_event.clear();
			_subtracted = false;
		}
		/**
		 * Returns the transverse momenta of the subtracted particles
		 *
		 * @return	vector of transverse momenta
		 */
		operator std::vector<double>(void)
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
		/**
		 * Returns the four-momenta of the subtracted particles
		 *
		 * @return	vector of four-momenta
		 */
		operator std::vector<snowmass_vector_t<double> >(void)
		{
			subtract_if_necessary();

			std::vector<snowmass_vector_t<double> > ret;

			for (std::vector<particle_t>::const_iterator iterator =
					 _event.begin();
				 iterator != _event.end(); iterator++) {
				snowmass_vector_t<double> p = iterator->momentum;

				p.perp() = iterator->momentum_perp_subtracted;
				p.set_lightlike_time();
				ret.push_back(p);
			}

			return ret;
		}
		/**
		 * Returns the area in the Voronoi diagram diagram occupied by
		 * a given particle
		 *
		 * @return	vector of area
		 */
		std::vector<double> particle_area(void)
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
		std::vector<std::set<size_t> > particle_incident(void)
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
	};

}
=======
	/**
	 * Add a new unsubtracted particle to the current event
	 *
	 * @param[in]	perp	transverse momentum
	 * @param[in]	pseudorapidity	pseudorapidity
	 * @param[in]	azimuth	azimuth
	 * @param[in]	reduced_particle_flow_id	reduced particle
	 * flow ID, between 0 and 2 (inclusive)
	 */
	void push_back_particle(
		const double perp, const double pseudorapidity,
		const double azimuth,
		const unsigned int reduced_particle_flow_id);
	/**
	 * Clears the list of unsubtracted particles
	 */
	void clear(void);
	/**
	 * Returns the transverse momenta of the subtracted particles
	 *
	 * @return	vector of transverse momenta
	 */
	operator std::vector<double>(void);
	/**
	 * Returns the area in the Voronoi diagram diagram occupied by
	 * a given particle
	 *
	 * @return	vector of area
	 */
	std::vector<double> particle_area(void);
	/**
	 * Returns the incident particles in the Delaunay diagram
	 * (particles that has a given particle as the nearest
	 * neighbor)
	 *
	 * @return	vector of sets of incident particles
	 * indices, using the original indexing
	 */
	std::vector<std::set<size_t> > particle_incident(void);
};
>>>>>>> 11c3b50... Split HF/Voronoi algorithm header and source

#endif
