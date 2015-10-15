#ifndef __GenericVoronoiAlgorithm_h__
#define __GenericVoronoiAlgorithm_h__

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

class GenericVoronoiAlgorithm {
protected:
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
		double momentum_perp_subtracted_unequalized;
		std::set<std::vector<particle_t>::iterator> incident;
		particle_t(math::PtEtaPhiELorentzVector p,
				   unsigned int i, double a = NAN,
				   double ps = NAN)
			: momentum(p), reduced_particle_flow_id(i), area(a),
			  momentum_perp_subtracted(ps),
			  momentum_perp_subtracted_unequalized(ps),
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
	static const size_t nfourier = 5;
protected:
	std::vector<double> _edge_pseudorapidity;
	std::vector<double> _cms_hcal_edge_pseudorapidity;
	std::vector<double> _cms_ecal_edge_pseudorapidity;
	std::pair<double, double> _equalization_threshold;
	bool _remove_nonpositive;
	event_t _event;
	//
	double _radial_distance_square_max;
	double _positive_bound_scale;
	std::vector<bool> _active;
	std::vector<std::pair<size_t, size_t> > _recombine;
	std::vector<std::vector<size_t> > _recombine_index;
	std::vector<std::vector<size_t> > _recombine_unsigned;
	std::vector<double> _recombine_tie;
	size_t _ncost;
	std::vector<size_t> _nblock_subtract;
	//
	bool _subtracted;
	boost::multi_array<double, 4> *_perp_fourier;
	std::vector<double> _feature;
protected:
	void initialize_geometry(void);
	void allocate(void);
	void deallocate(void);
	void event_fourier(void);
	void feature_extract(void);
	void voronoi_area_incident(void);
	virtual void subtract_momentum(void) = 0;
	void recombine_link(void);
	void lp_populate(void *lp_problem);
	void equalize(void);
	void remove_nonpositive(void);
	void subtract_if_necessary(void);
public:
	GenericVoronoiAlgorithm(
		const double dr_max = 0.4,
		const std::pair<double, double> equalization_threshold =
		std::pair<double, double>(5.0, 35.0),
		const bool remove_nonpositive = true);
	~GenericVoronoiAlgorithm(void);
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
	std::vector<double> subtracted_equalized_perp(void);
	std::vector<double> subtracted_unequalized_perp(void);
	/**
	 * Returns the area in the Voronoi diagram diagram occupied by
	 * a given particle
	 *
	 * @return	vector of area
	 */
	std::vector<double> particle_area(void);
	std::vector<std::set<size_t> > particle_incident(void);
	std::vector<double> perp_fourier(void);
	size_t nedge_pseudorapidity(void) const;
};

#endif
