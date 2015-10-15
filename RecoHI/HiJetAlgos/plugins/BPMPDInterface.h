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

namespace lp {

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

		void clear_row(void);
		void clear_column(void);
		void clear_coefficient(void);
		void to_csr(std::vector<int> &column_pointer,
					std::vector<int> &row_index,
					std::vector<double> &nonzero_value,
					const int index_start = 1);
	public:
		problem_t(const bool variable_named)
			: _optimized(false), _variable_named(variable_named)
		{
		}
		~problem_t(void)
		{
			clear();
		}
		void clear(void);
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
						   const double rhs);
		void push_back_row(const std::string &constraint_sense,
						   const double rhs);
		void push_back_column(const double objective,
							  const double lower_bound,
							  const double upper_bound);
		void push_back_coefficient(
			const int row, const int column, const double value);
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
		int ampl_solution_status(const int termination_code);
		void set_bpmpd_parameter(void);
		void append_constraint_sense_bound(
			std::vector<double> &lower_bound_bpmpd,
			std::vector<double> &upper_bound_bpmpd);
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
					  const double mf = 6.0, const size_t ma = 0);
		int optimize(void);
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
			std::vector<double> &variable_dual);
		friend class bpmpd_environment_t;
	};

}

#endif
