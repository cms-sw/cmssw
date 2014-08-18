#include "RecoHI/HiJetAlgos/interface/VoronoiAlgorithm.h"

extern "C" {

	extern struct ascal_s {
		double objnor;
		double rhsnor;
		double scdiff;
		int scpass;
		int scalmet;
	} ascal_;

	extern struct compl_s {
		double climit;
		double ccorr;
	} compl_;

	extern struct dims_s {
		int n;
		int n1;
		int m;
		int mn;
		int nz;
		int cfree;
		int pivotn;
		int denwin;
		int rfree;
	} dims_;

	extern struct drop_s {
		double tfixvar;
		double tfixslack;
		double slklim;
	} drop_;

	extern struct factor_s {
		double tpiv1;
		double tpiv2;
		double tabs;
		double trabs;
		double lam;
		double tfind;
		double order;
		double supdens;
	} factor_;

	extern struct initv_s {
		double prmin;
		double upmax;
		double dumin;
		int stamet;
		int safmet;
		int premet;
		int regul;
	} initv_;

	extern struct itref_s {
		double tresx;
		double tresy;
		int maxref;
	} itref_;

	extern struct mscal_s {
		double varadd;
		double slkadd;
		double scfree;
	} mscal_;

	extern struct numer_s {
		double tplus;
		double tzer;
	} numer_;

	extern struct param_s {
		double palpha;
		double dalpha;
	} param_;

	extern struct predc_s {
		double target;
		double tsmall;
		double tlarge;
		double center;
		double corstp;
		int mincc;
		int maxcc;
	} predc_;

	extern struct predp_s {
		double ccstop;
		double barset;
		double bargrw;
		double barmin;
		int mincor;
		int maxcor;
		int inibar;
	} predp_;

	extern struct setden_s {
		double maxdense;
		double densgap;
		int setlam;
		int denslen;
	} setden_;

	extern struct sprnod_s {
		int psupn;
		int ssupn;
		int maxsnz;
	} sprnod_;

	extern struct toler_s {
		double tsdir;
		double topt1;
		double topt2;
		double tfeas1;
		double tfeas2;
		double feas1;
		double feas2;
		double  pinfs;
		double dinfs;
		double inftol;
		int maxiter;
	} toler_;

	extern int solver_(
		double *obj, double *rhs, double *lbound, double *ubound,
		/* Size dims_.mn working arrays */
		double *diag, double *odiag,
		/* Size dims_.mn primal values */
		double *xs,
		/* Size dims_.mn working arrays */
		double *dxs, double *dxsn, double *up,
		/* Size dims_.mn dual residuals */
		double *dspr,
		/* Size dims_.mn working arrays */
		double *ddspr, double *ddsprn, double *dsup, double *ddsup,
		double *ddsupn,
		/* Size dims_.m dual values */
		double *dv,
		/* Size dims_.m working arrays */
		double *ddv, double *ddvn, double *prinf,
		/* Size dims_.mn working arrays */
		double *upinf, double *duinf, double *scale,
		/* Size dims_.cfree nonzero values */
		double *nonzeros,
		/* Size dims_.n working array */
		int *vartyp,
		/* Size dims_.m working array */
		int *slktyp,
		/* Size dims_.n1 column pointer */
		int *colpnt,
		/* Size dims_.mn working arrays */
		int *ecolpnt, int *count, int *vcstat, int *pivots,
		int *invprm, int *snhead, int *nodtyp, int *inta1,
		int *prehis,
		/* Size dims_.cfree row index */
		int *rowidx,
		/* Size dims_.rfree working array */
		int *rindex,
		/* Scalar termination code */
		int *code,
		/* Scalar optimum value */
		double *opt,
		int *iter, int *corect, int *fixn, int *dropn, int *fnzmax,
		int *fnzmin, double *addobj, double *bigbou,
		/* Practical +inf */
		double *big,
		int *ft);
}

namespace {

	class problem_t {
	public:
		static constexpr double infinity	= INFINITY;
		// These are equivalent to the constants used by IBM ILOG
		// CPLEX
		static const int minimize		=  1;
		static const int maximize		= -1;
		static const char less_equal	= 'L';
		static const char equal			= 'E';
		static const char greater_equal	= 'G';
		static const char range			= 'R';
	protected:
		bool _optimized;
		bool _variable_named;

		std::vector<double> _rhs;
		std::vector<char> _constraint_sense;
		std::vector<double> _range_value;
		std::vector<char *> _row_name;

		std::vector<double> _objective;
		std::vector<double> _lower_bound;
		std::vector<double> _upper_bound;
		std::vector<char *> _column_name;

		std::vector<int> _coefficient_row;
		std::vector<int> _coefficient_column;
		std::vector<double> _coefficient_value;

		void clear_row(void)
		{
			_rhs.clear();
			_constraint_sense.clear();
			for (std::vector<char *>::const_iterator iterator =
					 _row_name.begin();
				 iterator != _row_name.end(); iterator++) {
				delete [] *iterator;
			}
			_row_name.clear();
		}
		void clear_column(void)
		{
			_objective.clear();
			_lower_bound.clear();
			_upper_bound.clear();
			for (std::vector<char *>::const_iterator iterator =
					 _column_name.begin();
				 iterator != _column_name.end(); iterator++) {
				free(*iterator);
			}
			_column_name.clear();
		}
		void clear_coefficient(void)
		{
			_coefficient_row.clear();
			_coefficient_column.clear();
			_coefficient_value.clear();
		}
		void to_csr(std::vector<int> &column_pointer,
					std::vector<int> &row_index,
					std::vector<double> &nonzero_value,
					const int index_start = 1)
		{
			// Convert from coordinate (COO) storage into compressed
			// sparse row (CSR)

			std::vector<std::vector<std::pair<int, double> > >
				column_index(_objective.size(),
							 std::vector<std::pair<int, double> >());

			std::vector<int>::const_iterator iterator_row =
				_coefficient_row.begin();
			std::vector<int>::const_iterator iterator_column =
				_coefficient_column.begin();
			std::vector<double>::const_iterator iterator_value =
				_coefficient_value.begin();

			for (; iterator_value != _coefficient_value.end();
				 iterator_row++, iterator_column++, iterator_value++) {
				column_index[*iterator_column].push_back(
					std::pair<int, double>(
						// Conversion into Fortran indexing
						*iterator_row + index_start, *iterator_value));
			}

			for (std::vector<std::vector<std::pair<int, double> > >::
					 iterator iterator_outer = column_index.begin();
				 iterator_outer != column_index.end(); iterator_outer++) {
				// Conversion into Fortran indexing
				column_pointer.push_back(row_index.size() + index_start);
				std::sort(iterator_outer->begin(), iterator_outer->end());
				for (std::vector<std::pair<int, double> >::const_iterator
						 iterator_inner = iterator_outer->begin();
					 iterator_inner != iterator_outer->end();
					 iterator_inner++) {
					row_index.push_back(iterator_inner->first);
					nonzero_value.push_back(iterator_inner->second);
				}
			}
			// Conversion into Fortran indexing
			column_pointer.push_back(row_index.size() + index_start);
		}
	public:
		problem_t(const bool variable_named)
			: _optimized(false), _variable_named(variable_named)
		{
		}
		~problem_t(void)
		{
			clear();
		}
		void clear(void)
		{
			clear_row();
			clear_column();
			clear_coefficient();
		}
		virtual int populate(void) = 0;
		size_t nrow(void) const
		{
			return _rhs.size();
		}
		size_t ncolumn(void) const
		{
			return _objective.size();
		}
		void push_back_row(const char constraint_sense,
						   const double rhs)
		{
			_rhs.push_back(rhs);
			_constraint_sense.push_back(constraint_sense);

			if (_variable_named) {
				static const size_t name_length = 24;
				char *name = new char[name_length];

				snprintf(name, name_length, "c%llu",
						 static_cast<unsigned long long>(_rhs.size()));
				_row_name.push_back(name);
			}
		}
		void push_back_row(const std::string &constraint_sense,
						   const double rhs)
		{
			char cplex_sense;

			if (constraint_sense == "<=") {
				cplex_sense = 'L';
				push_back_row(rhs, cplex_sense);
			}
			else if (constraint_sense == "==") {
				cplex_sense = 'E';
				push_back_row(rhs, cplex_sense);
			}
			else if (constraint_sense == ">=") {
				cplex_sense = 'G';
				push_back_row(rhs, cplex_sense);
			}
			else {
				fprintf(stderr, "%s:%d: illegal sense (`%s')\n",
						__FILE__, __LINE__, constraint_sense.c_str());
			}
		}
		void push_back_column(const double objective,
							  const double lower_bound,
							  const double upper_bound)
		{
			_objective.push_back(objective);
			_lower_bound.push_back(lower_bound);
			_upper_bound.push_back(upper_bound);

			if (_variable_named) {
				static const size_t name_length = 24;
				char *name = new char[name_length];

				snprintf(name, name_length, "x%llu",
						 static_cast<unsigned long long>(
							_objective.size()));
				_column_name.push_back(name);
			}
		}
		void push_back_coefficient(
			const int row, const int column, const double value)
		{
			_coefficient_row.push_back(row);
			_coefficient_column.push_back(column);
			_coefficient_value.push_back(value);
		}
		virtual int optimize(void) = 0;
		int optimize_primal_simplex(void)
		{
			return optimize();
		}
		int optimize_dual_simplex(void)
		{
			return optimize();
		}
		int optimize_barrier(void)
		{
			return optimize();
		}
		int optimize_network(void)
		{
			return optimize();
		}
		int optimize_sifting(void)
		{
			return optimize();
		}
		int optimize_concurrent(void)
		{
			return optimize();
		}
		int optimize_deterministic_concurrent(void)
		{
			return optimize();
		}
		//
		virtual int solve(
			int &solution_status, double &objective_value,
			std::vector<double> &variable_primal,
			std::vector<double> &variable_dual,
			std::vector<double> &variable_slack_surplus,
			std::vector<double> &reduced_cost) = 0;
		int solve(
			double &objective_value,
			std::vector<double> &variable_primal,
			std::vector<double> &variable_dual,
			std::vector<double> &variable_slack_surplus,
			std::vector<double> &reduced_cost)
		{
			int solution_status;

			return solve(solution_status, objective_value,
						 variable_primal, variable_dual,
						 variable_slack_surplus, reduced_cost);
		}
		int solve(
			std::vector<double> &variable_primal,
			std::vector<double> &variable_dual,
			std::vector<double> &variable_slack_surplus,
			std::vector<double> &reduced_cost)
		{
			int solution_status;
			double objective_value;

			return solve(solution_status, objective_value,
						 variable_primal, variable_dual,
						 variable_slack_surplus, reduced_cost);
		}
		virtual int solve(
			int &solution_status, double &objective_value,
			std::vector<double> &variable_primal,
			std::vector<double> &variable_dual)
		{
			std::vector<double> variable_slack_surplus;
			std::vector<double> reduced_cost;

			return solve(solution_status, objective_value,
						 variable_primal, variable_dual,
						 variable_slack_surplus, reduced_cost);
		}
		int solve(
			std::vector<double> &variable_primal,
			std::vector<double> &variable_dual)
		{
			int solution_status;
			double objective_value;

			return solve(solution_status, objective_value, variable_primal,
						 variable_dual);
		}
		int solve(
			std::vector<double> &variable_primal)
		{
			std::vector<double> variable_dual;

			return solve(variable_primal, variable_dual);
		}
	};

	class environment_t {
	protected:
		std::vector<problem_t *> _problem;
	};

#ifndef BPMPD_INFINITY_BOUND
#define BPMPD_INFINITY_BOUND 1e+30
#endif // BPMPD_INFINITY_BOUND

#ifndef BPMPD_VERSION_STRING
#define BPMPD_VERSION_STRING "Version 2.11 (1996 December)"
#endif // BPMPD_VERSION_STRING

	class bpmpd_environment_t;

	class bpmpd_problem_t : public problem_t {
	public:
		static constexpr double infinity	= BPMPD_INFINITY_BOUND;
	protected:
		double _objective_sense;
		double _objective_value;
		std::vector<double> _variable_primal;
		std::vector<double> _variable_dual;
		int _solution_status;
	private:
		int ampl_solution_status(const int termination_code)
		{
			switch (termination_code) {
				// General memory limit (no solution)
			case -2:	return 520;
				// Memory limit during iterations
			case -1:	return 510;
				// No optimum
			case  1:	return 506;
			// Optimal solution
			case  2:	return   0;
				// Primal infeasible
			case  3:	return 503;
				// Dual infeasible
			case  4:	return 503;
			default:
				fprintf(stderr, "%s:%d: error: unknown termination code "
						"%d\n", __FILE__, __LINE__, termination_code);
				return 500;
			}
			fprintf(stderr, "%s:%d: %d\n", __FILE__, __LINE__,
					termination_code);
		}
		void set_bpmpd_parameter(void)
		{
			// Set the parameter as in subroutine BPMAIN

			setden_.maxdense = 0.15;
			setden_.densgap  = 3.00;
			setden_.setlam   = 0;
			factor_.supdens  = 250;
			setden_.denslen  = 10;

			mscal_.varadd    = 1e-12;
			mscal_.slkadd    = 1e+16;
			mscal_.scfree    = 1.1e-6;

			factor_.tpiv1    = 1e-3;
			factor_.tpiv2    = 1e-8;
			factor_.tabs     = 1e-12;
			factor_.trabs    = 1e-15;
			factor_.lam      = 1e-5;
			factor_.tfind    = 25;
			factor_.order    = 2;

			sprnod_.psupn    = 3;
			sprnod_.ssupn    = 4;
			sprnod_.maxsnz   = 9999999;

			toler_.tsdir     = 1e-16;
			toler_.topt1     = 1e-8;
			toler_.topt2     = 1e-16;
			toler_.tfeas1    = 1e-7;
			toler_.tfeas2    = 1e-7;
			toler_.feas1     = 1e-2;
			toler_.feas2     = 1e-2;
			toler_.pinfs     = 1e-6;
			toler_.dinfs     = 1e-6;
			toler_.inftol    = 1e+4;
			toler_.maxiter   = 99;

			numer_.tplus     = 1e-11;
			numer_.tzer      = 1e-35;

			itref_.tresx     = 1e-9;
			itref_.tresy     = 1e-9;
			itref_.maxref    = 5;

			ascal_.objnor    = 1e+2;
			ascal_.rhsnor    = 0;
			ascal_.scdiff    = 1;
			ascal_.scalmet   = 2;
			ascal_.scpass    = 5;

			predp_.ccstop    = 1.01;
			predp_.barset    = 2.50e-1;
			predp_.bargrw    = 1.00e+2;
			predp_.barmin    = 1.00e-10;
			predp_.mincor    = 1;
			predp_.maxcor    = 1;
			predp_.inibar    = 0;

			predc_.target    = 9e-2;
			predc_.tsmall    = 2e-1;
			predc_.tlarge    = 2e+1;
			predc_.center    = 5;
			predc_.corstp    = 1.01;
			predc_.mincc     = 0;
			predc_.maxcc     = 9;

			param_.palpha    = 0.999;
			param_.dalpha    = 0.999;

			drop_.tfixvar    = 1e-16;
			drop_.tfixslack  = 1e-16;
			drop_.slklim     = 1e-16;

			initv_.prmin     = 100;
			initv_.upmax     = 50000;
			initv_.dumin     = 100;
			initv_.stamet    = 2;
			initv_.safmet    = -3;
			initv_.premet    = 511;
			initv_.regul     = 0;

			compl_.climit    = 1;
			compl_.ccorr     = 1e-5;
		}
		void append_constraint_sense_bound(
			std::vector<double> &lower_bound_bpmpd,
			std::vector<double> &upper_bound_bpmpd)
		{
			for (std::vector<char>::const_iterator iterator =
					 _constraint_sense.begin();
				 iterator != _constraint_sense.end(); iterator++) {
				switch (*iterator) {
				case less_equal:
					lower_bound_bpmpd.push_back(-BPMPD_INFINITY_BOUND);
					upper_bound_bpmpd.push_back(0);
					break;
				case equal:
					lower_bound_bpmpd.push_back(0);
					upper_bound_bpmpd.push_back(0);
					break;
				case greater_equal:
					lower_bound_bpmpd.push_back(0);
					upper_bound_bpmpd.push_back(BPMPD_INFINITY_BOUND);
					break;
				case range:
					{
						const size_t index =
							iterator - _constraint_sense.begin();

						lower_bound_bpmpd.push_back(0);
						upper_bound_bpmpd.push_back(_range_value[index] -
													_rhs[index]);
					}
					break;
				}
			}
			lower_bound_bpmpd.reserve(dims_.mn);
			lower_bound_bpmpd.insert(lower_bound_bpmpd.end(),
									 _lower_bound.begin(),
									 _lower_bound.end());
			upper_bound_bpmpd.reserve(dims_.mn);
			upper_bound_bpmpd.insert(upper_bound_bpmpd.end(),
									 _upper_bound.begin(),
									 _upper_bound.end());
		}
	protected:
		int populate(void)
		{
			return 0;
		}
	public:
		bpmpd_problem_t(void)
			: problem_t(false), _objective_sense(1.0),
			  _objective_value(NAN)
		{
		}
		inline size_t nrow(void) const
		{
			return dims_.m;
		}
		inline size_t ncolumn(void) const
		{
			return dims_.n;
		}
		inline void set_objective_sense(int sense)
		{
			_objective_sense = sense;

			// This will be multiplied to the objective coefficients
			// (i.e. sign flip when sense = -1 for maximization)
		}
		inline void save(std::string filename)
		{
			// MPS save?
		}
		void push_back_column(const double objective,
							  const double lower_bound = 0,
							  const double upper_bound = BPMPD_INFINITY_BOUND)
		{
			problem_t::push_back_column(objective, lower_bound,
										upper_bound);
		}
		void set_size(const size_t nonzero_value_size,
					  const double mf = 6.0, const size_t ma = 0)
		{
			dims_.n = _objective.size();
			dims_.m = _rhs.size();
			dims_.mn = dims_.m + dims_.n;
			dims_.n1 = dims_.n + 1;

			dims_.nz = nonzero_value_size;

			// Adapted from the AMPL interface
			// http://www.netlib.org/ampl/solvers/bpmpd/

			const size_t rp_23 = 4 * dims_.m + 18 * dims_.mn;

			dims_.cfree = static_cast<int>(
				std::max(2.0, mf) * (rp_23 + dims_.nz)) +
				std::max(static_cast<size_t>(0), ma);
			dims_.rfree = ((dims_.cfree + rp_23) >> 1) + 11 * dims_.m +
				8 * dims_.n;
		}
		int optimize(void)
		{
			// Usually referred to as the variable "IA" for the CSR format
			std::vector<int> column_pointer;
			// Usually referred to as the variable "JA" for the CSR format
			std::vector<int> row_index;
			// Usually referred to as the variable "A" for the CSR format
			std::vector<double> nonzero_value;

			to_csr(column_pointer, row_index, nonzero_value);
			std::transform(_objective.begin(), _objective.end(),
						   _objective.begin(),
						   std::bind1st(std::multiplies<double>(),
										_objective_sense));

			// Try 1x, 2x, and 4x the default memory allocation
			for (size_t i = 0; i < 3; i++) {
				set_size(nonzero_value.size(), 6.0 * (1 << i));

				nonzero_value.resize(dims_.cfree);
				row_index.resize(dims_.cfree);

				set_bpmpd_parameter();

				std::vector<double> diag(dims_.mn, 0);
				std::vector<double> odiag(dims_.mn, 0);
				std::vector<double> dxs(dims_.mn, 0);
				std::vector<double> dxsn(dims_.mn, 0);
				std::vector<double> up(dims_.mn, 0);
				std::vector<double> dual_residual(dims_.mn, 0);
				std::vector<double> ddspr(dims_.mn, 0);
				std::vector<double> ddsprn(dims_.mn, 0);
				std::vector<double> dsup(dims_.mn, 0);
				std::vector<double> ddsup(dims_.mn, 0);
				std::vector<double> ddsupn(dims_.mn, 0);
				std::vector<double> ddv(dims_.m, 0);
				std::vector<double> ddvn(dims_.m, 0);
				std::vector<double> prinf(dims_.m, 0);
				std::vector<double> upinf(dims_.mn, 0);
				std::vector<double> duinf(dims_.mn, 0);
				std::vector<double> scale(dims_.mn, 0);
				std::vector<int> vartyp(dims_.n, 0);
				std::vector<int> slktyp(dims_.m, 0);
				std::vector<int> colpnt(dims_.n1, 0);
				std::vector<int> ecolpnt(dims_.mn, 0);
				std::vector<int> count(dims_.mn, 0);
				std::vector<int> vcstat(dims_.mn, 0);
				std::vector<int> pivots(dims_.mn, 0);
				std::vector<int> invprm(dims_.mn, 0);
				std::vector<int> snhead(dims_.mn, 0);
				std::vector<int> nodtyp(dims_.mn, 0);
				std::vector<int> inta1(dims_.mn, 0);
				std::vector<int> prehis(dims_.mn, 0);
				std::vector<int> rindex(dims_.rfree, 0);
				int termination_code;
				int iter;
				int correct;
				int fixn;
				int dropn;
				int fnzmax = 0;
				int fnzmin = -1;
				double addobj = 0;
				// Numerical limit of bounds to be ignored
				double bigbou = 1e+15;
				double infinity_copy = BPMPD_INFINITY_BOUND;
				int ft;

				_variable_primal.resize(dims_.mn);
				_variable_dual.resize(dims_.m);

				std::vector<double> lower_bound_bpmpd = _lower_bound;
				std::vector<double> upper_bound_bpmpd = _upper_bound;

				append_constraint_sense_bound(lower_bound_bpmpd,
											  upper_bound_bpmpd);

				solver_(&_objective[0], &_rhs[0], &lower_bound_bpmpd[0],
						&upper_bound_bpmpd[0], &diag[0], &odiag[0],
						&_variable_primal[0], &dxs[0], &dxsn[0], &up[0],
						&dual_residual[0], &ddspr[0], &ddsprn[0],
						&dsup[0], &ddsup[0], &ddsupn[0],
						&_variable_dual[0], &ddv[0], &ddvn[0], &prinf[0],
						&upinf[0], &duinf[0], &scale[0],
						&nonzero_value[0], &vartyp[0], &slktyp[0],
						&column_pointer[0], &ecolpnt[0], &count[0],
						&vcstat[0], &pivots[0], &invprm[0], &snhead[0],
						&nodtyp[0], &inta1[0], &prehis[0], &row_index[0],
						&rindex[0], &termination_code, &_objective_value,
						&iter, &correct, &fixn, &dropn, &fnzmax, &fnzmin,
						&addobj, &bigbou, &infinity_copy, &ft);

				_objective_value *= _objective_sense;
				_solution_status = ampl_solution_status(termination_code);
				if (termination_code != -2) {
					// No out-of-memory errors
					break;
				}
			}

			return _solution_status == 0 ? 0 : 1;
		}
		int solve(
			int &solution_status, double &objective_value,
			std::vector<double> &variable_primal,
			std::vector<double> &variable_dual,
			std::vector<double> &variable_slack_surplus,
			std::vector<double> &reduced_cost)
		{
			// This set of solution is not implemented yet (or readily
			// available from BPMPD)

			return 1;
		}
		int solve(
			int &solution_status, double &objective_value,
			std::vector<double> &variable_primal,
			std::vector<double> &variable_dual)
		{
			variable_primal = std::vector<double>(
				_variable_primal.begin(),
				_variable_primal.begin() + _objective.size());
			variable_dual = _variable_dual;

			return 0;
		}
		friend class bpmpd_environment_t;
	};

	class bpmpd_environment_t : public environment_t {
	public:
		bpmpd_environment_t(void)
		{
		}
		~bpmpd_environment_t(void)
		{
		}
		int set_verbose(void);
		int set_data_checking(void);
		inline std::string version_str(void) const
		{
			return BPMPD_VERSION_STRING;
		}
		bpmpd_problem_t problem(std::string name = "")
		{
			return bpmpd_problem_t();
		}
	};

}

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
		double ret;

		if (x >= -cody_waite_x_max && x <= cody_waite_x_max) {
			static const double inverse_two_pi =
				0.15915494309189534197;
			const double k = rint(x * inverse_two_pi);
			ret = ((x - (k * two_pi_0)) - k * two_pi_1) -
				k * two_pi_2;
		}
		else {
			long double sin_x;
			long double cos_x;

			sincosl(x, &sin_x, &cos_x);
			ret = (double)atan2l(sin_x, cos_x);
		}
		if (ret == -M_PI) {
			ret = M_PI;
		}

		return ret;
	}


	size_t pf_id_reduce(const int pf_id)
	{
		// Particle::pdgId_ PFCandidate::particleId_
		// PFCandidate::ParticleType Particle
		// 0           0  X          unknown, or dummy 
		// +211, -211  1  h          charged hadron 
		// +11, -11    2  e          electron 
		// +13, -13    3  mu         muon 
		// 22          4  gamma      photon 
		// 130         5  h0         neutral hadron 
		// 130         6  h_HF       hadronic energy in an HF tower 
		// 22          7  egamma_HF  electromagnetic energy in an HF tower

		if (pf_id == 4) {
			return 1;
		}
		else if (pf_id >= 5 && pf_id <= 7) {
			return 2;
		}

		return 0;
	}

}

		void VoronoiAlgorithm::initialize_geometry(void)
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
		void VoronoiAlgorithm::allocate(void)
		{
			_perp_fourier = new boost::multi_array<double, 4>(
				boost::extents[_edge_pseudorapidity.size() - 1]
				[nreduced_particle_flow_id][nfourier][2]);
		}
		void VoronoiAlgorithm::deallocate(void)
		{
			delete _perp_fourier;
		}
		void VoronoiAlgorithm::event_fourier(void)
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
		void VoronoiAlgorithm::feature_extract(void)
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
		void VoronoiAlgorithm::voronoi_area_incident(void)
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
		void VoronoiAlgorithm::subtract_momentum(void)
		{
			for (std::vector<particle_t>::iterator iterator =
					 _event.begin();
				 iterator != _event.end(); iterator++) {
				int predictor_index = -1;
				int interpolation_index = -1;
				double density = 0;
				double pred_0 = 0;

				for (size_t l = 1; l < _edge_pseudorapidity.size(); l++) {
					if (iterator->momentum.Eta() >=
						_edge_pseudorapidity[l - 1] &&
						iterator->momentum.Eta() <
						_edge_pseudorapidity[l]) {
						predictor_index = l - 1;
					}
				}

				for (size_t j = 0; j < nreduced_particle_flow_id; j++) {
				if (j == 2) {
					// HCAL
					for (size_t l = 1;
						 l < _cms_hcal_edge_pseudorapidity.size(); l++) {
						if (iterator->momentum.Eta() >=
							_cms_hcal_edge_pseudorapidity[l - 1] &&
							iterator->momentum.Eta() <
							_cms_hcal_edge_pseudorapidity[l]) {
							interpolation_index = l - 1;
						}
					}
				}
				else {
					// Tracks or ECAL clusters
					for (size_t l = 1;
						 l < _cms_ecal_edge_pseudorapidity.size(); l++) {
						if (iterator->momentum.Eta() >=
							_cms_ecal_edge_pseudorapidity[l - 1] &&
							iterator->momentum.Eta() <
							_cms_ecal_edge_pseudorapidity[l]) {
							interpolation_index = l - 1;
						}
					}
				}

				if (predictor_index >= 0 && interpolation_index >= 0) {
					// Calculate the aggregated prediction and
					// interpolation for the pseudorapidity segment

					const double azimuth = iterator->momentum.Phi();
					const float (*p)[2][82] =
#ifdef STANDALONE
						ue_predictor_pf[j][predictor_index]
#else // STANDALONE
						ue->ue_predictor_pf[j][predictor_index]
#endif // STANDALONE
						;
					double pred = 0;

					for (size_t l = 0; l < nfourier; l++) {
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

				if (std::isfinite(iterator->area) && density >= 0) {
					// Subtract the PF candidate by density times
					// Voronoi cell area
					iterator->momentum_perp_subtracted =
						iterator->momentum.Pt() -
						density * iterator->area;
				}
				else {
					iterator->momentum_perp_subtracted =
						iterator->momentum.Pt();
				}
				iterator->momentum_perp_subtracted_unequalized =
					iterator->momentum_perp_subtracted;
			}
		}
		void VoronoiAlgorithm::recombine_link(void)
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
		void VoronoiAlgorithm::lp_populate(void *lp_problem)
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
		void VoronoiAlgorithm::equalize(void)
		{
			bpmpd_problem_t lp_problem = reinterpret_cast<bpmpd_environment_t *>(_lp_environment)->problem();

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
		void VoronoiAlgorithm::remove_nonpositive(void)
		{
			for (std::vector<particle_t>::iterator iterator =
					 _event.begin();
				 iterator != _event.end(); iterator++) {
				iterator->momentum_perp_subtracted = std::max(
					0.0, iterator->momentum_perp_subtracted);
			}
		}
		void VoronoiAlgorithm::subtract_if_necessary(void)
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

		VoronoiAlgorithm::VoronoiAlgorithm(
			const double dr_max,
			const bool isRealData,
			const bool isCalo,
			const std::pair<double, double> equalization_threshold,
			const bool remove_nonpositive)
			: _remove_nonpositive(remove_nonpositive),
			  _equalization_threshold(equalization_threshold),
			  _radial_distance_square_max(dr_max * dr_max),
			  _positive_bound_scale(0.2),
			  _subtracted(false),
			  ue(NULL)
		{
			initialize_geometry();
			ue = new UECalibration(isRealData, isCalo);
			static const size_t nedge_pseudorapidity = 15 + 1;
			static const double edge_pseudorapidity[nedge_pseudorapidity] = {
				-5.191, -2.650, -2.043, -1.740, -1.479, -1.131, -0.783, -0.522, 0.522, 0.783, 1.131, 1.479, 1.740, 2.043, 2.650, 5.191
			};

			_edge_pseudorapidity = std::vector<double>(
				edge_pseudorapidity,
				edge_pseudorapidity + nedge_pseudorapidity);
			allocate();
		}

		VoronoiAlgorithm::~VoronoiAlgorithm(void)
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
		void VoronoiAlgorithm::push_back_particle(
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
		void VoronoiAlgorithm::clear(void)
		{
			_event.clear();
			_subtracted = false;
		}
		/**
		 * Returns the transverse momenta of the subtracted particles
		 *
		 * @return	vector of transverse momenta
		 */
		std::vector<double> VoronoiAlgorithm::subtracted_equalized_perp(void)
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
		std::vector<double> VoronoiAlgorithm::subtracted_unequalized_perp(void)
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
		std::vector<double> VoronoiAlgorithm::particle_area(void)
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
		std::vector<std::set<size_t> > VoronoiAlgorithm::particle_incident(void)
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
		std::vector<double> VoronoiAlgorithm::perp_fourier(void)
		{
			subtract_if_necessary();

			return std::vector<double>(
				_perp_fourier->data(),
				_perp_fourier->data() +
				_perp_fourier->num_elements());
		}
		size_t VoronoiAlgorithm::nedge_pseudorapidity(void) const
		{
			return _edge_pseudorapidity.size();
		}
