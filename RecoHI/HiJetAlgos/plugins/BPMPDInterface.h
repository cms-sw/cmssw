#ifndef __BPMPDInterface_h__
#define __BPMPDInterface_h__

#include <vector>
#include <string>
#include <algorithm>

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
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-const-variable"
		static constexpr double infinity	= INFINITY;
		// These are equivalent to the constants used by IBM ILOG
		// CPLEX
		static const int minimize		=  1;
		static const int maximize		= -1;
		static const char less_equal	= 'L';
		static const char equal			= 'E';
		static const char greater_equal	= 'G';
		static const char range			= 'R';
#pragma clang diagnostic pop
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
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-const-variable"
		static constexpr double infinity	= BPMPD_INFINITY_BOUND;
#pragma clang diagnostic pop
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

#endif
