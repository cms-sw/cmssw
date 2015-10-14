#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "BPMPDInterface.h"

namespace lp {

	void problem_t::clear_row(void)
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

	void problem_t::clear_column(void)
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

	void problem_t::clear_coefficient(void)
	{
		_coefficient_row.clear();
		_coefficient_column.clear();
		_coefficient_value.clear();
	}
	void problem_t::to_csr(std::vector<int> &column_pointer,
						   std::vector<int> &row_index,
						   std::vector<double> &nonzero_value,
						   const int index_start)
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

	void problem_t::clear(void)
	{
		clear_row();
		clear_column();
		clear_coefficient();
	}

	void problem_t::push_back_row(const char constraint_sense,
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

	void problem_t::push_back_row(const std::string &constraint_sense,
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
			edm::LogError("BPMPDInterface") << "illegal sense (`" << constraint_sense << "')" << std::endl;
		}
	}

	void problem_t::push_back_column(const double objective,
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

	void problem_t::push_back_coefficient(
		const int row, const int column, const double value)
	{
		_coefficient_row.push_back(row);
		_coefficient_column.push_back(column);
		_coefficient_value.push_back(value);
	}

	int bpmpd_problem_t::ampl_solution_status(const int termination_code)
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
			edm::LogError("BPMPDInterface") << "unknown termination code " << termination_code << std::endl;
			return 500;
		}
	}

	void bpmpd_problem_t::set_bpmpd_parameter(void)
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

	void bpmpd_problem_t::append_constraint_sense_bound(
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

	void bpmpd_problem_t::set_size(const size_t nonzero_value_size,
								   const double mf, const size_t ma)
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

	int bpmpd_problem_t::optimize(void)
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

	int bpmpd_problem_t::solve(
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

}
